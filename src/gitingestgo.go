package main

import (
	"bufio"
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/fatih/color"
	"github.com/mr-tron/base58/base58"
	"github.com/pkoukk/tiktoken-go"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/viper"
)

// AppConfig corresponds to the structure in settings.json
type AppConfig struct {
	DefaultIgnorePatterns []string `mapstructure:"default_ignore_patterns"`
	MaxDirectoryDepth     int      `mapstructure:"max_directory_depth"`
	MaxFiles              int      `mapstructure:"max_files"`
	MaxTotalSizeBytes     int64    `mapstructure:"max_total_size_bytes"`
	DefaultIgnoreFile     string   `mapstructure:"default_ignore_file"`
}

// Colored log helpers
var (
	warningLog = color.New(color.FgYellow).PrintfFunc()
	infoLog    = color.New(color.FgCyan).PrintfFunc()
	errorLog   = color.New(color.FgRed).PrintfFunc()
	skipLog    = color.New(color.FgMagenta).SprintfFunc() // For final summary
)

// FileInfo holds essential file data
type FileInfo struct {
	Path    string
	Content string
	Size    int64
}

func main() {
	// CLI flags
	inputPtr := flag.String("d", "", "Directory (or file) to process")
	outputPtr := flag.String("o", "", "Output file name")
	maxFileSizePtr := flag.Int("s", 10, "Max file size in KB (defaults to 10KB)")
	flag.Parse()

	if *inputPtr == "" {
		log.Fatal("error: you must provide an input directory/file using -d <path>")
	}
	if *outputPtr == "" {
		log.Fatal("error: you must provide an output file name using -o <filename>")
	}

	// 1. Load config from settings.json
	config, err := loadConfig("settings.json")
	if err != nil {
		errorLog("error loading config: %v\n", err)
		os.Exit(1)
	}

	// 2. Parse user query (merges config’s ignore patterns, etc.)
	query, err := parseQuery(*inputPtr, *maxFileSizePtr, config)
	if err != nil {
		errorLog("error parsing query: %v\n", err)
		os.Exit(1)
	}

	// 3. Ingest data (scan, build tree, combine file content)
	summary, tree, content, skipped, err := IngestFromQuery(query)
	if err != nil {
		errorLog("error during ingestion: %v\n", err)
		os.Exit(1)
	}

	// 4. Write results to output file
	outputContent := "Directory structure:\n" + tree + "\n" + content
	if writeErr := os.WriteFile(*outputPtr, []byte(outputContent), 0644); writeErr != nil {
		errorLog("error writing output to file '%s': %v\n", *outputPtr, writeErr)
		os.Exit(1)
	}

	// 5. Display success
	infoLog("Successfully processed path: %s\n", *inputPtr)
	infoLog("Output written to: %s\n", *outputPtr)
	fmt.Println(summary)

	// 6. Display a rich summary of skipped resources
	printSkippedSummary(skipped)
}

/*
loadConfig reads settings.json using Viper.
*/
func loadConfig(configFile string) (AppConfig, error) {
	var config AppConfig

	viper.SetConfigFile(configFile)
	err := viper.ReadInConfig()
	if err != nil {
		return config, err
	}
	err = viper.Unmarshal(&config)
	if err != nil {
		return config, err
	}
	return config, nil
}

/*
parseQuery merges:
- Command-line inputs
- The config from settings.json
- Returns a map with all relevant scanning info
*/
func parseQuery(input string, maxFileSizeKB int, config AppConfig) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	if input == "" {
		return nil, errors.New("no input path specified")
	}
	absPath, err := filepath.Abs(input)
	if err != nil {
		return nil, fmt.Errorf("invalid local path: %w", err)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("the specified path does not exist: %s", absPath)
	}

	// Populate essential fields
	result["local_path"] = absPath
	result["slug"] = filepath.Base(absPath)
	result["subpath"] = "/"

	// Generate short ID
	idBytes := sha256.Sum256([]byte(absPath))
	result["id"] = base58.Encode(idBytes[:16])

	// Convert KB to bytes
	result["max_file_size"] = int64(maxFileSizeKB * 1024)

	// Merge config’s default ignore patterns
	ignorePatterns := make([]string, 0, len(config.DefaultIgnorePatterns))
	ignorePatterns = append(ignorePatterns, config.DefaultIgnorePatterns...)

	result["ignore_patterns"] = ignorePatterns

	// Additional config fields if you need them:
	result["max_directory_depth"] = config.MaxDirectoryDepth
	result["max_files"] = config.MaxFiles
	result["max_total_size_bytes"] = config.MaxTotalSizeBytes
	result["default_ignore_file"] = config.DefaultIgnoreFile

	return result, nil
}

/*
IngestFromQuery:
 1. Scans the directory
 2. Formats file contents
 3. Batch-counts tokens
 4. Builds summary
    => Also returns a slice of skipped resources to display rich output later
*/
func IngestFromQuery(query map[string]interface{}) (
	summary string,
	tree string,
	filesContent string,
	skipped []string,
	err error,
) {
	localPath := query["local_path"].(string)
	ignorePatterns := query["ignore_patterns"].([]string)
	maxFileSize := query["max_file_size"].(int64)

	// 1. Scan (files, dirTree, and a list of skipped paths)
	files, dirTree, skippedResources, scanErr := scanLocalDirectory(localPath, ignorePatterns, maxFileSize)
	if scanErr != nil {
		return "", "", "", nil, fmt.Errorf("scan error: %w", scanErr)
	}

	// 2. Format
	formattedContent := formatFilesContent(files)

	// 3. Batch token counting
	enc, encErr := tiktoken.GetEncoding("cl100k_base")
	if encErr != nil {
		return "", "", "", nil, fmt.Errorf("failed to get encoding: %w", encErr)
	}
	tokenCount, tokenErr := batchCountTokens(enc, files, dirTree)
	if tokenErr != nil {
		return "", "", "", nil, fmt.Errorf("token counting error: %w", tokenErr)
	}

	// 4. Summary
	slug, _ := query["slug"].(string)
	summaryText := createSummary(slug, "", "", "/", len(files), tokenCount)

	return summaryText, dirTree, formattedContent, skippedResources, nil
}

/*
scanLocalDirectory uses improved isIgnored to skip directories mentioned
in ignore patterns (like build/, dist/, etc.). It now also collects
skipped paths in a slice for later reporting.
*/
func scanLocalDirectory(
	basePath string,
	ignorePatterns []string,
	maxFileSize int64,
) ([]FileInfo, string, []string, error) {

	var files []FileInfo
	var dirStructure strings.Builder
	var skippedPaths []string
	var mu sync.Mutex
	var wg sync.WaitGroup

	bar := progressbar.NewOptions(-1,
		progressbar.OptionSetDescription("Scanning files/directories"),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() {
			fmt.Println()
		}),
	)
	defer bar.Finish()

	fi, err := os.Lstat(basePath)
	if err != nil {
		return nil, "", nil, fmt.Errorf("unable to get file info for '%s': %w", basePath, err)
	}

	if fi.IsDir() {
		err = filepath.Walk(basePath, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				warningLog("warning: error during Walk at '%s': %v\n", path, walkErr)
				return walkErr
			}
			relPath, _ := filepath.Rel(basePath, path)
			if relPath == "." {
				return nil
			}

			_ = bar.Add(1) // progress increment

			// Check if path is ignored
			if isIgnored(relPath, info.IsDir(), ignorePatterns) {
				// Store this path for later summary
				mu.Lock()
				skippedPaths = append(skippedPaths, relPath)
				mu.Unlock()

				if info.IsDir() {
					// Skip entire directory
					return filepath.SkipDir
				}
				return nil
			}

			// Build ASCII tree
			if info.IsDir() {
				appendToTree(&dirStructure, relPath, true)
			} else if isTextFile(path) && info.Size() <= maxFileSize {
				appendToTree(&dirStructure, relPath, false)

				wg.Add(1)
				go func(p string, size int64, rPath string) {
					defer wg.Done()
					content, fileErr := readFileLimited(p, maxFileSize)
					if fileErr != nil {
						warningLog("warning: reading file '%s' failed: %v\n", p, fileErr)
						return
					}
					mu.Lock()
					files = append(files, FileInfo{Path: rPath, Content: content, Size: size})
					mu.Unlock()
				}(path, info.Size(), relPath)
			}
			return nil
		})
	} else {
		_ = bar.Add(1)
		// Single file scenario
		if isTextFile(basePath) && fi.Size() <= maxFileSize {
			content, readErr := readFileLimited(basePath, maxFileSize)
			if readErr == nil {
				files = append(files, FileInfo{
					Path:    filepath.Base(basePath),
					Content: content,
					Size:    fi.Size(),
				})
				dirStructure.WriteString(filepath.Base(basePath) + "\n")
			} else {
				warningLog("warning: reading single file '%s' failed: %v\n", basePath, readErr)
			}
		}
	}

	wg.Wait()

	if err != nil {
		return files, dirStructure.String(), skippedPaths, fmt.Errorf("error walking directory '%s': %w", basePath, err)
	}
	return files, dirStructure.String(), skippedPaths, nil
}

/*
isIgnored checks if relPath matches any pattern from settings.json.
*/
func isIgnored(relPath string, isDir bool, patterns []string) bool {
	normalizedPath := filepath.ToSlash(relPath)
	for _, p := range patterns {
		p = strings.TrimSpace(p)
		if p == "" || strings.HasPrefix(p, "#") {
			continue
		}
		if matchPath(normalizedPath, p, isDir) {
			return true
		}
	}
	return false
}

/*
matchPath handles directory patterns (ending with /), wildcard files (*.exe),
and exact matches. This is a simplified approach but covers typical cases.
*/
func matchPath(path, pattern string, isDir bool) bool {
	pattern = strings.TrimSpace(pattern)

	// Directory pattern => ends with '/'
	if strings.HasSuffix(pattern, "/") {
		dirName := strings.TrimSuffix(pattern, "/")
		if isDir {
			if path == dirName || strings.HasPrefix(path, dirName+"/") {
				return true
			}
		} else {
			if strings.HasPrefix(path, dirName+"/") {
				return true
			}
		}
		return false
	}

	// If pattern has '*', treat it as wildcard
	if strings.Contains(pattern, "*") {
		baseName := filepath.Base(path)
		matched, _ := filepath.Match(pattern, baseName)
		return matched
	}

	// Otherwise exact match
	return path == pattern
}

// isTextFile uses heuristics
func isTextFile(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		warningLog("warning: could not open file '%s': %v\n", filename, err)
		return false
	}
	defer f.Close()

	bom := make([]byte, 4)
	if _, err := f.Read(bom); err == nil {
		if isUnicodeBOM(bom) {
			return true
		}
	}
	f.Seek(0, io.SeekStart)

	buf := make([]byte, 1024)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		warningLog("warning: could not read file '%s': %v\n", filename, err)
		return false
	}
	for _, b := range buf[:n] {
		if b == 0 || (b < 32 && b != 9 && b != 10 && b != 13) {
			return false
		}
	}
	return true
}

func isUnicodeBOM(b []byte) bool {
	return (b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF)
}

// readFileLimited reads up to maxSize
func readFileLimited(path string, maxSize int64) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return "", err
	}
	if fi.Size() > maxSize {
		return "", fmt.Errorf("file '%s' exceeds max file size limit", path)
	}

	var sb strings.Builder
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		sb.WriteString(scanner.Text())
		sb.WriteString("\n")
	}
	if sErr := scanner.Err(); sErr != nil {
		return "", sErr
	}
	return sb.String(), nil
}

/*
appendToTree: simplified ASCII tree. If you need accurate ├ vs. └, track sibling indexes.
*/
func appendToTree(builder *strings.Builder, relPath string, isDir bool) {
	depth := strings.Count(relPath, string(os.PathSeparator))
	if depth == 0 {
		if isDir {
			builder.WriteString(relPath + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(relPath + "\n")
		}
		return
	}
	prefix := strings.Repeat("│   ", depth-1)
	name := filepath.Base(relPath)
	if isDir {
		builder.WriteString(prefix + "└── " + name + string(os.PathSeparator) + "\n")
	} else {
		builder.WriteString(prefix + "└── " + name + "\n")
	}
}

/*
formatFilesContent arranges text files with a heading
and prioritizes README.md if present.
*/
func formatFilesContent(files []FileInfo) string {
	var sb strings.Builder
	separator := strings.Repeat("=", 48) + "\n"

	// Place README.md first if it exists
	for i, file := range files {
		if strings.ToLower(file.Path) == "readme.md" {
			sb.WriteString(separator)
			sb.WriteString(fmt.Sprintf("File: %s\n", file.Path))
			sb.WriteString(separator)
			sb.WriteString(file.Content + "\n\n")
			files = append(files[:i], files[i+1:]...)
			break
		}
	}

	// The rest of the files
	for _, file := range files {
		sb.WriteString(separator)
		sb.WriteString(fmt.Sprintf("File: %s\n", file.Path))
		sb.WriteString(separator)
		sb.WriteString(file.Content + "\n\n")
	}
	return sb.String()
}

/*
batchCountTokens uses the *tiktoken.Tiktoken encoder to compute approximate token usage.
*/
func batchCountTokens(enc *tiktoken.Tiktoken, files []FileInfo, dirTree string) (string, error) {
	totalTokens := 0

	// Encode directory structure
	dirTokens := enc.Encode(dirTree, nil, nil)
	totalTokens += len(dirTokens)

	// Encode each file’s content
	for _, f := range files {
		toks := enc.Encode(f.Content, nil, nil)
		totalTokens += len(toks)
	}

	// Convert numeric token count to a user-friendly string
	switch {
	case totalTokens > 1_000_000:
		return fmt.Sprintf("%.1fM", float64(totalTokens)/1_000_000), nil
	case totalTokens > 1_000:
		return fmt.Sprintf("%.1fk", float64(totalTokens)/1_000), nil
	default:
		return fmt.Sprintf("%d", totalTokens), nil
	}
}

/*
createSummary shows repo name, file count, etc.
*/
func createSummary(repoName, branch, commit, subpath string, fileCount int, tokenCount string) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Repository: %s\n", repoName))
	sb.WriteString(fmt.Sprintf("Files analyzed: %d\n", fileCount))

	if subpath != "" && subpath != "/" {
		sb.WriteString(fmt.Sprintf("Subpath: %s\n", subpath))
	}
	if commit != "" {
		sb.WriteString(fmt.Sprintf("Commit: %s\n", commit))
	} else if branch != "" && branch != "main" && branch != "master" {
		sb.WriteString(fmt.Sprintf("Branch: %s\n", branch))
	}
	sb.WriteString(fmt.Sprintf("Estimated tokens: %s", tokenCount))
	return sb.String()
}

/*
printSkippedSummary prints all skipped resources after scanning, in color.
*/
func printSkippedSummary(skipped []string) {
	if len(skipped) == 0 {
		infoLog("\nNo resources were skipped based on your ignore patterns!\n")
		return
	}

	// Title
	fmt.Println()
	color.New(color.FgMagenta, color.Bold).Println("Skipped Resources:")
	for _, s := range skipped {
		fmt.Printf("  %s\n", skipLog("%s", s))
	}
	fmt.Println()
}

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
	"regexp"
	"strings"
	"sync"

	"github.com/mr-tron/base58/base58"
	"github.com/pkoukk/tiktoken-go"
	"github.com/spf13/viper"
)

// AppConfig maps to the fields in settings.json
type AppConfig struct {
	DefaultIgnorePatterns []string `mapstructure:"default_ignore_patterns"`
	MaxDirectoryDepth     int      `mapstructure:"max_directory_depth"`
	MaxFiles              int      `mapstructure:"max_files"`
	MaxTotalSizeBytes     int64    `mapstructure:"max_total_size_bytes"`
	DefaultIgnoreFile     string   `mapstructure:"default_ignore_file"`
}

// FileInfo holds essential file data, including path, content, and size.
type FileInfo struct {
	Path    string
	Content string
	Size    int64
}

func main() {
	/*
	   Command-line Flags:

	   -d string : Directory (or file) to scan
	   -o string : Output file to write combined output
	   -s int    : Max file size in KB (default 10)
	   -i string : Include patterns (comma-separated)
	   -e string : Exclude patterns (comma-separated)

	   Example usage:
	   go run main.go -d ./your_project -o output.txt -s 20 -i "*.go" -e "*.md"
	*/
	inputPtr := flag.String("d", "", "Directory (or file) to process")
	outputPtr := flag.String("o", "", "Output file name")
	maxFileSizePtr := flag.Int("s", 10, "Max file size in KB")
	includePatternsPtr := flag.String("i", "", "Include patterns (comma-separated)")
	excludePatternsPtr := flag.String("e", "", "Exclude patterns (comma-separated)")
	flag.Parse()

	// Basic validation of required flags
	if *inputPtr == "" {
		log.Fatal("error: you must provide an input directory/file using -d <path>")
	}
	if *outputPtr == "" {
		log.Fatal("error: you must provide an output file name using -o <filename>")
	}

	// 1. Load configuration via Viper
	config, err := loadConfig("settings.json") // or "appsettings.json", etc.
	if err != nil {
		log.Fatalf("error: failed to load config file: %v", err)
	}

	// 2. Parse user query (integrates advanced reasoning to filter out unneeded content)
	query, err := ParseQuery(
		*inputPtr,
		*maxFileSizePtr, // user-provided maxFileSize in KB
		false,           // fromWeb
		*includePatternsPtr,
		*excludePatternsPtr,
		config,
	)
	if err != nil {
		log.Fatalf("error parsing query: %v", err)
	}

	// 3. Ingest data (scan, build tree, combine file content)
	summary, tree, content, err := IngestFromQuery(query)
	if err != nil {
		log.Fatalf("error during ingestion: %v", err)
	}

	// 4. Write results to the specified output file
	outputContent := "Directory structure:\n" + tree + "\n" + content
	if writeErr := os.WriteFile(*outputPtr, []byte(outputContent), 0644); writeErr != nil {
		log.Fatalf("error writing output to file '%s': %v", *outputPtr, writeErr)
	}

	// 5. Display success message and summary to the console
	fmt.Printf("Successfully processed path: %s\n", *inputPtr)
	fmt.Printf("Output written to: %s\n", *outputPtr)
	fmt.Println(summary)
}

/*
loadConfig reads the specified JSON config file using Viper,
then unmarshals it into an AppConfig struct.
*/
func loadConfig(configFile string) (AppConfig, error) {
	var config AppConfig

	viper.SetConfigFile(configFile)
	if err := viper.ReadInConfig(); err != nil {
		return config, err
	}

	if err := viper.Unmarshal(&config); err != nil {
		return config, err
	}

	return config, nil
}

/*
ParseQuery orchestrates the input parsing:
- Checks if input is a local path vs. a web URL (not implemented here).
- Loads default ignore patterns from config; merges with user-supplied exclude patterns.
- Creates a short "id" from hashing the absolute path.
- Returns a map that consolidates all relevant query metadata.
*/
func ParseQuery(
	input string,
	maxFileSizeKB int,
	fromWeb bool,
	includePatternsStr, excludePatternsStr string,
	config AppConfig,
) (map[string]interface{}, error) {

	result := make(map[string]interface{})

	if fromWeb {
		// For demonstration, we return an explicit error here if fromWeb is true
		return nil, errors.New("web URL processing is not implemented for CLI in this example")
	}

	// Validate local path and convert to absolute
	absPath, err := filepath.Abs(input)
	if err != nil {
		return nil, fmt.Errorf("invalid local path: %w", err)
	}

	// Basic input validation: ensure path exists
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("the specified path does not exist: %s", absPath)
	}

	// Populate result with key-value pairs
	result["local_path"] = absPath
	result["slug"] = filepath.Base(absPath) // short project slug
	result["subpath"] = "/"

	// Generate a short ID from path-based hashing
	idBytes := sha256.Sum256([]byte(absPath))
	result["id"] = base58.Encode(idBytes[:16]) // Keep first 16 bytes for brevity

	// Use the config’s default ignore patterns
	defaultIgnorePatterns := config.DefaultIgnorePatterns

	// Merge user-supplied exclude patterns
	var finalIgnorePatterns []string
	finalIgnorePatterns = append(finalIgnorePatterns, defaultIgnorePatterns...)

	if excludePatternsStr != "" {
		excludePatterns := parsePatterns(excludePatternsStr)
		finalIgnorePatterns = append(finalIgnorePatterns, excludePatterns...)
	}
	result["ignore_patterns"] = finalIgnorePatterns

	// Include patterns override certain ignores
	if includePatternsStr != "" {
		includePatterns := parseIncludePatterns(includePatternsStr)
		result["include_patterns"] = includePatterns
		finalIgnorePatterns = overrideIgnorePatterns(finalIgnorePatterns, includePatterns)
		result["ignore_patterns"] = finalIgnorePatterns
	}

	// Convert KB to bytes for maxFileSize
	result["max_file_size"] = int64(maxFileSizeKB * 1024)

	// Also carry over other config values if needed
	result["default_ignore_file"] = config.DefaultIgnoreFile
	result["max_directory_depth"] = config.MaxDirectoryDepth
	result["max_files"] = config.MaxFiles
	result["max_total_size_bytes"] = config.MaxTotalSizeBytes

	return result, nil
}

/*
IngestFromQuery executes the main scanning logic by:
- Reading .gitignore (if exists) from the local path
- Merging ignore patterns
- Scanning the path
- Returning summary, ASCII tree, and combined file contents
*/
func IngestFromQuery(query map[string]interface{}) (
	summary string,
	tree string,
	filesContent string,
	err error,
) {
	localPath := query["local_path"].(string)

	// Optional: read .gitignore if it exists in the directory
	var gitIgnorePatterns []string
	gitIgnoreFile := filepath.Join(localPath, query["default_ignore_file"].(string))
	if _, statErr := os.Stat(gitIgnoreFile); statErr == nil {
		if patterns, parseErr := parseIgnoreFile(gitIgnoreFile); parseErr == nil {
			gitIgnorePatterns = append(gitIgnorePatterns, patterns...)
		} else {
			log.Printf("info: error parsing '%s': %v", gitIgnoreFile, parseErr)
		}
	}

	// Merge user-supplied ignore patterns with .gitignore
	ignorePatterns := []string{}
	if userIgnore, ok := query["ignore_patterns"].([]string); ok {
		ignorePatterns = append(ignorePatterns, userIgnore...)
	}
	ignorePatterns = append(ignorePatterns, gitIgnorePatterns...)

	// Always ignore .git/
	ignorePatterns = append(ignorePatterns, ".git/")

	maxFileSize := query["max_file_size"].(int64)
	files, dirTree, scanErr := scanLocalDirectory(localPath, ignorePatterns, maxFileSize)
	if scanErr != nil {
		return "", "", "", fmt.Errorf("scan error: %w", scanErr)
	}

	// Build the combined file content output
	formattedContent := formatFilesContent(files)

	// Count tokens to provide an estimate
	fullTextForTokenCount := dirTree + formattedContent
	tokenCount, tokenErr := countTokens(fullTextForTokenCount)
	if tokenErr != nil {
		return "", "", "", fmt.Errorf("token counting error: %w", tokenErr)
	}

	// Gather metadata for summary
	repoName, _ := query["slug"].(string)
	branch, _ := query["branch"].(string)
	commit, _ := query["commit"].(string)
	subpath, _ := query["subpath"].(string)

	summaryText := createSummary(repoName, branch, commit, subpath, len(files), tokenCount)

	return summaryText, dirTree, formattedContent, nil
}

// -------------------------------------------------------------------
// Below is the original scanning and helper logic, largely unchanged.
// -------------------------------------------------------------------

func parseIgnoreFile(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("unable to open ignore file: %w", err)
	}
	defer file.Close()

	var patterns []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		patterns = append(patterns, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading ignore file: %w", err)
	}

	return patterns, nil
}

func matchesPattern(path string, pattern string, isDir bool) bool {
	negate := false
	if strings.HasPrefix(pattern, "!") {
		negate = true
		pattern = pattern[1:]
	}

	if strings.HasPrefix(pattern, "/") {
		pattern = pattern[1:]
	}

	if strings.Contains(pattern, "**/") {
		pattern = strings.ReplaceAll(pattern, "**/", "(.*/)?")
	}

	if strings.HasSuffix(pattern, "/") {
		if isDir {
			pattern = strings.TrimSuffix(pattern, "/") + "(/.*)?"
		} else {
			return false
		}
	}

	pattern = "^" + strings.ReplaceAll(pattern, ".", "\\.")
	pattern = strings.ReplaceAll(pattern, "*", ".*")
	pattern += "$"

	matched, err := regexp.MatchString(pattern, path)
	if err != nil {
		log.Printf("warning: invalid pattern encountered: %v", err)
		return false
	}
	return matched == !negate
}

func isTextFile(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		log.Printf("warning: could not open file '%s': %v", filename, err)
		return false
	}
	defer f.Close()

	// Check BOM
	bom := make([]byte, 4)
	if _, err := f.Read(bom); err == nil {
		if isUnicodeBOM(bom) {
			return true
		}
	}

	_, _ = f.Seek(0, io.SeekStart)

	buf := make([]byte, 1024)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		log.Printf("warning: could not read from file '%s': %v", filename, err)
		return false
	}
	for _, b := range buf[:n] {
		if b == 0 || (b < 32 && b != 9 && b != 10 && b != 13) {
			return false
		}
	}
	return true
}

func isUnicodeBOM(bom []byte) bool {
	return (bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF) ||
		(bom[0] == 0xFE && bom[1] == 0xFF) ||
		(bom[0] == 0xFF && bom[1] == 0xFE) ||
		(bom[0] == 0x00 && bom[1] == 0x00 && bom[2] == 0xFE && bom[3] == 0xFF) ||
		(bom[0] == 0xFF && bom[1] == 0xFE && bom[2] == 0x00 && bom[3] == 0x00)
}

func readFileLimited(path string, maxFileSize int64) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open file '%s': %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to get file info '%s': %w", path, err)
	}
	if fi.Size() > maxFileSize {
		return "", fmt.Errorf("file '%s' exceeds max file size limit", path)
	}

	var builder strings.Builder
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		builder.WriteString(scanner.Text())
		builder.WriteString("\n")
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error scanning file '%s': %w", path, err)
	}
	return builder.String(), nil
}

func appendToTree(builder *strings.Builder, relPath string, basePath string, isDir bool) {
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
	isLast := isLastElement(basePath, relPath)
	name := filepath.Base(relPath)

	if isLast {
		if isDir {
			builder.WriteString(prefix + "└── " + name + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(prefix + "└── " + name + "\n")
		}
	} else {
		if isDir {
			builder.WriteString(prefix + "├── " + name + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(prefix + "├── " + name + "\n")
		}
	}
}

func isLastElement(basePath, relPath string) bool {
	parts := strings.Split(relPath, string(os.PathSeparator))
	if len(parts) < 2 {
		return true
	}

	parentDir := filepath.Join(basePath, strings.Join(parts[:len(parts)-1], string(os.PathSeparator)))
	currentBase := parts[len(parts)-1]

	entries, err := os.ReadDir(parentDir)
	if err != nil {
		log.Printf("warning: could not read parent directory '%s': %v", parentDir, err)
		return true
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return len(names) > 0 && names[len(names)-1] == currentBase
}

func formatFilesContent(files []FileInfo) string {
	var contentBuilder strings.Builder
	separator := strings.Repeat("=", 48) + "\n"

	// Place README.md first, if it exists
	for i, file := range files {
		if strings.ToLower(file.Path) == "readme.md" {
			contentBuilder.WriteString(separator)
			contentBuilder.WriteString(fmt.Sprintf("File: %s\n", file.Path))
			contentBuilder.WriteString(separator)
			contentBuilder.WriteString(file.Content + "\n\n")

			files = append(files[:i], files[i+1:]...)
			break
		}
	}

	for _, file := range files {
		contentBuilder.WriteString(separator)
		contentBuilder.WriteString(fmt.Sprintf("File: %s\n", file.Path))
		contentBuilder.WriteString(separator)
		contentBuilder.WriteString(file.Content + "\n\n")
	}
	return contentBuilder.String()
}

func countTokens(text string) (string, error) {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return "", fmt.Errorf("failed to get encoding: %w", err)
	}

	totalTokens := len(enc.Encode(text, nil, nil))
	switch {
	case totalTokens > 1_000_000:
		return fmt.Sprintf("%.1fM", float64(totalTokens)/1_000_000), nil
	case totalTokens > 1_000:
		return fmt.Sprintf("%.1fk", float64(totalTokens)/1_000), nil
	default:
		return fmt.Sprintf("%d", totalTokens), nil
	}
}

func createSummary(repoName, branch, commit, subpath string, fileCount int, tokenCount string) string {
	var summary strings.Builder
	summary.WriteString(fmt.Sprintf("Repository: %s\n", repoName))
	summary.WriteString(fmt.Sprintf("Files analyzed: %d\n", fileCount))

	if subpath != "" && subpath != "/" {
		summary.WriteString(fmt.Sprintf("Subpath: %s\n", subpath))
	}
	if commit != "" {
		summary.WriteString(fmt.Sprintf("Commit: %s\n", commit))
	} else if branch != "" && branch != "main" && branch != "master" {
		summary.WriteString(fmt.Sprintf("Branch: %s\n", branch))
	}

	summary.WriteString(fmt.Sprintf("Estimated tokens: %s", tokenCount))
	return summary.String()
}

func parsePatterns(patterns string) []string {
	var parsedPatterns []string
	for _, p := range strings.Split(patterns, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			parsedPatterns = append(parsedPatterns, p)
		}
	}
	return parsedPatterns
}

func parseIncludePatterns(pattern string) []string {
	parts := strings.Split(strings.TrimSpace(pattern), ",")
	var patterns []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if strings.HasPrefix(part, "./") {
			part = part[2:]
		} else if strings.HasPrefix(part, "/") {
			part = part[1:]
		}
		if strings.HasSuffix(part, "/") {
			patterns = append(patterns, part+"**")
		} else {
			patterns = append(patterns, part)
		}
	}
	return patterns
}

func overrideIgnorePatterns(ignorePatterns []string, includePatterns []string) []string {
	var newIgnorePatterns []string
	for _, ignorePattern := range ignorePatterns {
		shouldIgnore := true
		for _, includePattern := range includePatterns {
			matched, _ := filepath.Match(includePattern, ignorePattern)
			if matched {
				shouldIgnore = false
				break
			}
		}
		if shouldIgnore {
			newIgnorePatterns = append(newIgnorePatterns, ignorePattern)
		}
	}
	return newIgnorePatterns
}

/*
scanLocalDirectory scans the directory recursively (or processes a single file),
respecting ignore patterns, maximum file-size limits, etc.
It returns a list of FileInfo and a formatted directory tree string.
*/
func scanLocalDirectory(
	basePath string,
	ignorePatterns []string,
	maxFileSize int64,
) ([]FileInfo, string, error) {

	var files []FileInfo
	var dirStructure strings.Builder
	var mu sync.Mutex
	var wg sync.WaitGroup

	fileInfo, err := os.Lstat(basePath)
	if err != nil {
		return nil, "", fmt.Errorf("unable to get file info for '%s': %w", basePath, err)
	}

	isSymlinkToDir := false
	if fileInfo.Mode()&os.ModeSymlink != 0 {
		realPath, err := filepath.EvalSymlinks(basePath)
		if err != nil {
			return nil, "", fmt.Errorf("failed to evaluate symlink for '%s': %w", basePath, err)
		}
		realFileInfo, err := os.Stat(realPath)
		if err != nil {
			return nil, "", fmt.Errorf("failed to stat symlink target '%s': %w", realPath, err)
		}
		isSymlinkToDir = realFileInfo.IsDir()
	}

	// If it’s a directory (or symlink to directory), traverse
	if isSymlinkToDir || fileInfo.IsDir() {
		err = filepath.Walk(basePath, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				log.Printf("warning: error during filepath.Walk at '%s': %v", path, walkErr)
				return walkErr
			}

			if info.Mode()&os.ModeSymlink != 0 {
				log.Printf("info: skipping symlinked directory '%s'", path)
				return filepath.SkipDir
			}

			relPath, _ := filepath.Rel(basePath, path)
			if relPath == "." {
				return nil
			}

			// Check ignore patterns
			for _, pattern := range ignorePatterns {
				if matchesPattern(relPath, pattern, info.IsDir()) {
					if info.IsDir() {
						log.Printf("info: skipping directory '%s' due to ignore pattern '%s'", relPath, pattern)
						return filepath.SkipDir
					}
					log.Printf("info: skipping file '%s' due to ignore pattern '%s'", relPath, pattern)
					return nil
				}
			}

			if info.IsDir() {
				appendToTree(&dirStructure, relPath, basePath, true)
			} else if isTextFile(path) && info.Size() <= maxFileSize {
				appendToTree(&dirStructure, relPath, basePath, false)
				wg.Add(1)
				go func(p string, size int64) {
					defer wg.Done()
					content, fileErr := readFileLimited(p, maxFileSize)
					if fileErr != nil {
						log.Printf("warning: reading file '%s' failed: %v", p, fileErr)
						return
					}
					mu.Lock()
					files = append(files, FileInfo{Path: relPath, Content: content, Size: size})
					mu.Unlock()
				}(path, info.Size())
			}
			return nil
		})

	} else {
		// Single file
		if isTextFile(basePath) && fileInfo.Size() <= maxFileSize {
			content, readErr := readFileLimited(basePath, maxFileSize)
			if readErr == nil {
				files = append(files, FileInfo{
					Path:    filepath.Base(basePath),
					Content: content,
					Size:    fileInfo.Size(),
				})
				dirStructure.WriteString(filepath.Base(basePath) + "\n")
			} else {
				log.Printf("warning: reading single file '%s' failed: %v", basePath, readErr)
			}
		}
	}

	wg.Wait()

	if err != nil {
		return files, dirStructure.String(), fmt.Errorf("error walking directory '%s': %w", basePath, err)
	}

	return files, dirStructure.String(), nil
}

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

// AppConfig corresponds to the structure in settings.json.
// It includes default ignore patterns and various limits for scanning.
type AppConfig struct {
	// DefaultIgnorePatterns is a list of directory/file patterns
	// that should be skipped (e.g., build/, node_modules/, etc.).
	DefaultIgnorePatterns []string `mapstructure:"default_ignore_patterns"`

	// MaxDirectoryDepth sets how deep the recursive scan goes.
	MaxDirectoryDepth int `mapstructure:"max_directory_depth"`

	// MaxFiles is the maximum number of files that can be processed.
	MaxFiles int `mapstructure:"max_files"`

	// MaxTotalSizeBytes is the maximum combined size of all processed files.
	MaxTotalSizeBytes int64 `mapstructure:"max_total_size_bytes"`

	// DefaultIgnoreFile is the default ignore file name (e.g., .gitignore).
	DefaultIgnoreFile string `mapstructure:"default_ignore_file"`
}

// Colored log helpers for enriched console output.
var (
	warningLog = color.New(color.FgYellow).PrintfFunc()
	infoLog    = color.New(color.FgCyan).PrintfFunc()
	errorLog   = color.New(color.FgRed).PrintfFunc()
	skipLog    = color.New(color.FgMagenta).PrintfFunc() // Used for final summary of skipped items
)

// FileInfo holds essential metadata and content for a single file.
type FileInfo struct {
	// Path is the relative path of the file from the scanned directory.
	Path string

	// Content is the text content of the file.
	Content string

	// Size is the file size in bytes.
	Size int64
}

// main is the entry point for the CLI application.
// It parses flags, loads configuration, scans the directory or file,
// and outputs results to a file and console.
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

// loadConfig reads the specified configFile using the Viper library,
// unmarshals it into an AppConfig struct, and returns it.
//
//	configFile: The path to the JSON or YAML config file
//
// Returns:
//
//	(AppConfig, error): The loaded configuration or an error if loading fails.
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

// parseQuery merges command-line flags (e.g., maxFileSizeKB) with the
// loaded config (e.g., DefaultIgnorePatterns) into a single map.
//
//	input: The path to scan (file or directory).
//	maxFileSizeKB: Maximum file size in KB, passed as a CLI flag.
//	config: The AppConfig loaded from settings.json.
//
// Returns:
//
//	map[string]interface{}: A generic map containing all relevant scanning info
//	error: If the path is invalid or does not exist.
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

// IngestFromQuery orchestrates the entire ingestion process,
// including scanning directories, building a directory tree,
// reading file contents, counting tokens, and generating a summary.
//
//	query: A map with scanning info (local_path, ignore_patterns, etc.)
//
// Returns:
//
//	summary       (string): A human-readable summary (e.g., repo name, token count).
//	tree          (string): The ASCII representation of the directory structure.
//	filesContent  (string): The concatenated file contents (with headers).
//	skipped       ([]string): A list of skipped files/directories for final reporting.
//	err           (error): Any error encountered during scanning or token counting.
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

// scanLocalDirectory recursively walks through the specified basePath,
// skipping any directories/files that match ignorePatterns.
//
//	basePath       (string): The absolute path to a directory or file.
//	ignorePatterns ([]string): Patterns (e.g., "node_modules/", "*.exe") to skip.
//	maxFileSize    (int64): The maximum file size in bytes allowed.
//
// Returns:
//
//	files          ([]FileInfo): List of successfully read files with content.
//	dirStructure   (string): ASCII directory tree representation.
//	skippedPaths   ([]string): Accumulated list of skipped file/directory paths.
//	err            (error): Any error encountered during the walk or file reads.
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
		// Traverse directory tree
		err = filepath.Walk(basePath, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				warningLog("warning: error during Walk at '%s': %v\n", path, walkErr)
				return walkErr
			}
			relPath, _ := filepath.Rel(basePath, path)
			if relPath == "." {
				return nil
			}

			_ = bar.Add(1) // increment progress

			// Check if path is ignored
			if isIgnored(relPath, info.IsDir(), ignorePatterns) {
				// Store this path for final summary
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

				// Read file concurrently
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
		// Single file scenario
		_ = bar.Add(1)
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

// isIgnored checks if relPath matches any pattern from the provided patterns
// and returns true if it should be skipped.
//
//	relPath (string): The path relative to the base directory.
//	isDir   (bool):   Whether the path is a directory.
//	patterns([]string): The ignore patterns from config.
//
// Returns:
//
//	bool: true if the path is matched by an ignore pattern, otherwise false.
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

// matchPath handles directory patterns (ending with /), wildcard files (*.exe),
// and exact matches.
//
//	path    (string): normalized relative path to check (forward slashes).
//	pattern (string): current ignore pattern (e.g., build/, *.exe).
//	isDir   (bool):   true if the path is a directory.
//
// Returns:
//
//	bool: true if path matches the pattern, otherwise false.
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

// isTextFile performs a heuristic check to see if the file is likely text.
// It reads up to 1KB of the file to find null bytes or unusual control chars,
// which typically indicate a binary file.
//
//	filename (string): The path to the file.
//
// Returns:
//
//	bool: true if it passes text heuristics, false if it seems binary or unreadable.
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

// isUnicodeBOM checks if the provided 4 bytes correspond to a UTF-8 BOM.
//
//	b ([]byte): The first 4 bytes of a file.
//
// Returns:
//
//	bool: true if b is the UTF-8 BOM signature, otherwise false.
func isUnicodeBOM(b []byte) bool {
	return (b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF)
}

// readFileLimited reads the file content up to maxSize bytes.
//
//	path (string): The path to the file.
//	maxSize (int64): The maximum allowed file size in bytes.
//
// Returns:
//
//	string: The file content if within size limits.
//	error:  If the file is too large or cannot be read.
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

// appendToTree builds a simplified ASCII directory tree.
//
//	builder (*strings.Builder): The output accumulator for the tree representation.
//	relPath (string): The relative path to append (e.g., subdirectory/file).
//	isDir   (bool):   Whether the path is a directory.
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

// formatFilesContent arranges all file contents with a header. If a README.md
// is found, it is placed at the top. Each file is separated by a clear delimiter.
//
//	files ([]FileInfo): The list of FileInfo structs with path and content.
//
// Returns:
//
//	string: The combined text for all files, with ASCII headers.
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

// batchCountTokens encodes the directory tree and each file’s content to estimate
// the total token usage using tiktoken.
//
//	enc (*tiktoken.Tiktoken): The encoder for token counting.
//	files ([]FileInfo):       The files read from the directory.
//	dirTree (string):         The ASCII directory tree.
//
// Returns:
//
//	(string, error): A user-friendly string (e.g., "1.2k") if successful,
//	                 or an error if encoding fails.
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

// createSummary compiles a brief textual report about the scanning operation,
// including the number of files analyzed and approximate token count.
//
//	repoName (string): A short name or slug for the repository or project.
//	branch   (string): The branch name if any.
//	commit   (string): The commit SHA if any.
//	subpath  (string): A subdirectory path if scanning only a sub-tree.
//	fileCount(int):    Number of files that were successfully processed.
//	tokenCount(string): The user-friendly token count string.
//
// Returns:
//
//	string: A final summary line that can be printed to the console.
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

// printSkippedSummary prints all skipped resources after the scan has completed.
// This is called at the end of main for a richer final output.
//
//	skipped ([]string): A list of directory/file paths that were ignored
//	                    due to the configured ignore patterns.
func printSkippedSummary(skipped []string) {
	if len(skipped) == 0 {
		infoLog("\nNo resources were skipped based on your ignore patterns!\n")
		return
	}

	// Title
	fmt.Println()
	skipLog("Skipped Resources: %d\n", len(skipped))
	// for _, s := range skipped {
	// 	fmt.Printf("  %s\n", skipLog("%s", s))
	// }
	fmt.Println()
}

package main

import (
	"bufio"
	"crypto/sha256"
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
)

// --- Colored log helpers for convenience ---
var (
	warningLog = color.New(color.FgYellow).PrintfFunc()
	infoLog    = color.New(color.FgCyan).PrintfFunc()
	errorLog   = color.New(color.FgRed).PrintfFunc()
)

// FileInfo holds essential file data, including path, content, and size.
type FileInfo struct {
	Path    string
	Content string
	Size    int64
}

func main() {
	// Command-line flags
	inputPtr := flag.String("d", "", "Directory (or file) to process")
	outputPtr := flag.String("o", "", "Output file name")
	maxFileSizePtr := flag.Int("s", 10, "Max file size in KB")
	flag.Parse()

	if *inputPtr == "" {
		log.Fatal("error: you must provide an input directory/file using -d <path>")
	}
	if *outputPtr == "" {
		log.Fatal("error: you must provide an output file name using -o <filename>")
	}

	// Parse user query (simplified)
	query, err := parseQuery(*inputPtr, *maxFileSizePtr)
	if err != nil {
		errorLog("error parsing query: %v\n", err)
		os.Exit(1)
	}

	// Main ingestion: scan directory, format output, count tokens
	summary, tree, content, err := IngestFromQuery(query)
	if err != nil {
		errorLog("error during ingestion: %v\n", err)
		os.Exit(1)
	}

	// Write the directory structure + file content to the specified output file
	outputContent := "Directory structure:\n" + tree + "\n" + content
	if writeErr := os.WriteFile(*outputPtr, []byte(outputContent), 0644); writeErr != nil {
		errorLog("error writing output to file '%s': %v\n", *outputPtr, writeErr)
		os.Exit(1)
	}

	// Success messages
	infoLog("Successfully processed path: %s\n", *inputPtr)
	infoLog("Output written to: %s\n", *outputPtr)
	fmt.Println(summary)
}

/*
parseQuery orchestrates basic input logic:
- Ensures the path exists
- Converts it to absolute
- Returns a map of query metadata (e.g., local_path, max_file_size, ignore_patterns)
*/
func parseQuery(input string, maxFileSizeKB int) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	absPath, err := filepath.Abs(input)
	if err != nil {
		return nil, fmt.Errorf("invalid local path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("the specified path does not exist: %s", absPath)
	}

	result["local_path"] = absPath
	result["slug"] = filepath.Base(absPath) // e.g., "myproject"
	result["subpath"] = "/"

	// Generate a short ID from path-based hashing
	idBytes := sha256.Sum256([]byte(absPath))
	result["id"] = base58.Encode(idBytes[:16]) // keep it shorter

	// Convert KB to bytes
	result["max_file_size"] = int64(maxFileSizeKB * 1024)

	// Simplified ignore pattern (override in real code as needed)
	result["ignore_patterns"] = []string{".git/"}

	return result, nil
}

/*
IngestFromQuery handles:
- Scanning the local directory for text files
- Building an ASCII directory tree
- Formatting file contents
- Counting tokens in batches
*/
func IngestFromQuery(query map[string]interface{}) (string, string, string, error) {
	localPath := query["local_path"].(string)
	ignorePatterns := query["ignore_patterns"].([]string)
	maxFileSize := query["max_file_size"].(int64)

	// 1. Scan local directory (files + tree)
	files, dirTree, err := scanLocalDirectory(localPath, ignorePatterns, maxFileSize)
	if err != nil {
		return "", "", "", fmt.Errorf("scan error: %w", err)
	}

	// 2. Format file contents (placing README.md first, etc.)
	formattedContent := formatFilesContent(files)

	// 3. Batch token counting with the correct type: *tiktoken.Tiktoken
	enc, encErr := tiktoken.GetEncoding("cl100k_base") // or EncodingForModel("...")
	if encErr != nil {
		return "", "", "", fmt.Errorf("failed to get encoding: %w", encErr)
	}

	tokenCount, tokenErr := batchCountTokens(enc, files, dirTree)
	if tokenErr != nil {
		return "", "", "", fmt.Errorf("token counting error: %w", tokenErr)
	}

	// 4. Summary
	slug, _ := query["slug"].(string)
	summaryText := createSummary(slug, "", "", "/", len(files), tokenCount)

	return summaryText, dirTree, formattedContent, nil
}

/*
batchCountTokens takes:
- A *tiktoken.Tiktoken encoder
- A slice of FileInfo
- The ASCII directory tree

It encodes each piece of text (dir tree + file contents) to count tokens in batches.
*/
func batchCountTokens(enc *tiktoken.Tiktoken, files []FileInfo, dirTree string) (string, error) {
	totalTokens := 0

	// Encode directory tree
	dirTokens := enc.Encode(dirTree, nil, nil)
	totalTokens += len(dirTokens)

	// Encode each file's content
	for _, f := range files {
		fileTokens := enc.Encode(f.Content, nil, nil)
		totalTokens += len(fileTokens)
	}

	// Convert numeric token count to friendly string
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
scanLocalDirectory scans:
1. A directory (recursively) using filepath.Walk
2. Or a single file if the base path is not a directory

We integrate a progress bar (indeterminate) for user feedback.
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

	// Create an indeterminate progress bar (total = -1)
	bar := progressbar.NewOptions(-1,
		progressbar.OptionSetDescription("Scanning files/directories"),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowCount(),
		progressbar.OptionOnCompletion(func() {
			fmt.Println() // New line after finishing
		}),
	)
	defer bar.Finish()

	fileInfo, err := os.Lstat(basePath)
	if err != nil {
		return nil, "", fmt.Errorf("unable to get file info for '%s': %w", basePath, err)
	}

	// If it's a directory
	if fileInfo.IsDir() {
		err = filepath.Walk(basePath, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				warningLog("warning: error during Walk at '%s': %v\n", path, walkErr)
				return walkErr
			}

			relPath, _ := filepath.Rel(basePath, path)
			if relPath == "." {
				return nil
			}

			// Increment the progress bar for each file or directory visited
			_ = bar.Add(1)

			// Check ignore patterns
			for _, pattern := range ignorePatterns {
				if matchesPattern(relPath, pattern, info.IsDir()) {
					if info.IsDir() {
						infoLog("info: skipping directory '%s' (pattern: '%s')\n", relPath, pattern)
						return filepath.SkipDir
					}
					infoLog("info: skipping file '%s' (pattern: '%s')\n", relPath, pattern)
					return nil
				}
			}

			// Build ASCII directory tree
			if info.IsDir() {
				appendToTree(&dirStructure, relPath, basePath, true)
			} else if isTextFile(path) && info.Size() <= maxFileSize {
				appendToTree(&dirStructure, relPath, basePath, false)

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
					files = append(files, FileInfo{
						Path:    rPath,
						Content: content,
						Size:    size,
					})
					mu.Unlock()
				}(path, info.Size(), relPath)
			}

			return nil
		})
	} else {
		// Single file scenario
		_ = bar.Add(1)
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
				warningLog("warning: reading single file '%s' failed: %v\n", basePath, readErr)
			}
		}
	}

	wg.Wait()

	if err != nil {
		return files, dirStructure.String(), fmt.Errorf("error walking directory '%s': %w", basePath, err)
	}
	return files, dirStructure.String(), nil
}

// matchesPattern checks if the current path matches an ignore pattern.
func matchesPattern(path, pattern string, isDir bool) bool {
	// Very simplified example. Adjust to your real .gitignore logic.
	if strings.HasSuffix(pattern, "/") && !isDir {
		return false
	}
	if strings.Contains(path, strings.TrimSuffix(pattern, "/")) {
		return true
	}
	return false
}

// isTextFile uses heuristics to detect if a file is likely text.
func isTextFile(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		warningLog("warning: could not open file '%s': %v\n", filename, err)
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
		warningLog("warning: could not read file '%s': %v\n", filename, err)
		return false
	}

	for _, b := range buf[:n] {
		// Null bytes or unusual control chars typically indicate binary
		if b == 0 || (b < 32 && b != 9 && b != 10 && b != 13) {
			return false
		}
	}
	return true
}

// isUnicodeBOM checks for a UTF-8 BOM signature.
func isUnicodeBOM(b []byte) bool {
	return (b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF)
}

// readFileLimited reads the file up to maxSize bytes.
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
appendToTree builds the ASCII directory tree. This is a simplified version
that doesn't check if an item is the last sibling. Adjust if you need
accurate ├ vs. └ branching.
*/
func appendToTree(builder *strings.Builder, relPath, basePath string, isDir bool) {
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
	// Simplified: treat everything as 'last' in this example
	if isDir {
		builder.WriteString(prefix + "└── " + name + string(os.PathSeparator) + "\n")
	} else {
		builder.WriteString(prefix + "└── " + name + "\n")
	}
}

/*
formatFilesContent arranges text files to display their content with a
section header. It also places README.md at the top if it exists.
*/
func formatFilesContent(files []FileInfo) string {
	var sb strings.Builder
	separator := strings.Repeat("=", 48) + "\n"

	// Place README.md first if present
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

	// Append all other files
	for _, file := range files {
		sb.WriteString(separator)
		sb.WriteString(fmt.Sprintf("File: %s\n", file.Path))
		sb.WriteString(separator)
		sb.WriteString(file.Content + "\n\n")
	}
	return sb.String()
}

/*
createSummary forms a short textual summary of the scanning operation:
- Repo/slug name
- # of files
- Subpath, if any
- Estimated token count
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

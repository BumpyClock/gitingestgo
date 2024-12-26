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
	"regexp"
	"strings"
	"sync"

	"github.com/mr-tron/base58/base58"
	"github.com/pkoukk/tiktoken-go"
)

const (
	// MaxFileSize represents the maximum size of a file (in bytes) that will be processed.
	MaxFileSize = 10 * 1024 * 1024 // 10MB

	// MaxDirectoryDepth represents the maximum depth of directories that will be scanned.
	MaxDirectoryDepth = 20

	// MaxFiles represents the maximum number of files that will be processed.
	MaxFiles = 10000

	// MaxTotalSizeBytes represents the maximum total size of files (in bytes) that will be processed.
	MaxTotalSizeBytes = 500 * 1024 * 1024 // 500MB

	// DefaultIgnoreFile is the default name of the ignore file to use if no custom ignore patterns are provided.
	DefaultIgnoreFile = ".gitignore"
)

// FileInfo represents a file with its content and metadata.
type FileInfo struct {
	Path    string
	Content string
	Size    int64
}

// parseGitIgnore reads a .gitignore file and returns a slice of ignore patterns.
func parseGitIgnore(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var patterns []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}
		patterns = append(patterns, line)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return patterns, nil
}

// matchesPattern checks if a path matches a given .gitignore pattern.
func matchesPattern(path string, pattern string, isDir bool) bool {
	// Handle negation patterns
	negate := false
	if strings.HasPrefix(pattern, "!") {
		negate = true
		pattern = pattern[1:]
	}

	// Handle absolute paths in .gitignore
	if strings.HasPrefix(pattern, "/") {
		pattern = pattern[1:]
	}

	// Handle **/ for recursive directory matching
	if strings.Contains(pattern, "**/") {
		pattern = strings.ReplaceAll(pattern, "**/", "(.*/)?")
	}

	// Handle */ for directory matching at current level
	if strings.HasSuffix(pattern, "/") {
		if isDir {
			pattern = strings.TrimSuffix(pattern, "/") + "(/.*)?"
		} else {
			return false // Pattern is for directories but path is a file
		}
	}

	// Convert the .gitignore pattern to a regular expression
	pattern = "^" + strings.ReplaceAll(pattern, ".", "\\.")
	pattern = strings.ReplaceAll(pattern, "*", ".*")
	pattern = pattern + "$"

	matched, err := regexp.MatchString(pattern, path)
	if err != nil {
		return false // Invalid pattern
	}

	return matched == !negate
}

// scanDirectory recursively scans a directory, respecting ignore patterns and limits.
func scanDirectory(basePath string, ignorePatterns []string, maxFileSize int64) ([]FileInfo, string, error) {
	var files []FileInfo
	var dirStructure strings.Builder
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Check if the path is a symlink to a directory
	fileInfo, err := os.Lstat(basePath)
	if err != nil {
		return nil, "", fmt.Errorf("error getting file info: %v", err)
	}

	isSymlinkToDir := false
	if fileInfo.Mode()&os.ModeSymlink != 0 {
		realPath, err := filepath.EvalSymlinks(basePath)
		if err != nil {
			return nil, "", fmt.Errorf("error evaluating symlink: %v", err)
		}
		realFileInfo, err := os.Stat(realPath)
		if err != nil {
			return nil, "", fmt.Errorf("error getting real file info: %v", err)
		}
		isSymlinkToDir = realFileInfo.IsDir()
	}

	// if symlink to directory or just a normal directory, traverse it
	if isSymlinkToDir || fileInfo.IsDir() {
		err = filepath.Walk(basePath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err // Handle errors encountered during traversal.
			}

			// Prevent recursing into symlinked directories to avoid infinite loops.
			if info.Mode()&os.ModeSymlink != 0 {
				return filepath.SkipDir
			}

			relPath, _ := filepath.Rel(basePath, path)
			if relPath == "." {
				return nil // Skip the root directory itself.
			}

			// Check ignore patterns.
			for _, pattern := range ignorePatterns {
				if matchesPattern(relPath, pattern, info.IsDir()) {
					if info.IsDir() {
						return filepath.SkipDir
					}
					return nil
				}
			}

			if info.IsDir() {
				// Append directory to tree structure.
				appendToTree(&dirStructure, relPath, basePath, true)
			} else if isTextFile(path) && info.Size() <= maxFileSize {
				// Append file to tree structure.
				appendToTree(&dirStructure, relPath, basePath, false)
				// Process files concurrently.
				wg.Add(1)
				go func(path string, size int64) {
					defer wg.Done()
					if content, err := readFileContent(path, maxFileSize); err == nil {
						mu.Lock()
						files = append(files, FileInfo{Path: relPath, Content: content, Size: size})
						mu.Unlock()
					}
				}(path, info.Size())
			}

			return nil
		})
	} else if isTextFile(basePath) && fileInfo.Size() <= maxFileSize {
		// If it's a single file, process it directly.
		if content, err := readFileContent(basePath, maxFileSize); err == nil {
			files = append(files, FileInfo{Path: filepath.Base(basePath), Content: content, Size: fileInfo.Size()})
			dirStructure.WriteString(filepath.Base(basePath) + "\n")
		}
	}

	wg.Wait() // Wait for all file processing goroutines to complete.

	return files, dirStructure.String(), err
}

// isTextFile checks if a file is likely a text file based on its content.
func isTextFile(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		return false
	}
	defer f.Close()

	// Check for Unicode BOM at the beginning of the file
	bom := make([]byte, 4)
	if _, err := f.Read(bom); err == nil {
		if isUnicodeBOM(bom) {
			return true
		}
	}

	// Reset file offset after checking BOM
	f.Seek(0, io.SeekStart)

	buf := make([]byte, 1024)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		return false
	}

	// Check for presence of null bytes or control characters
	for _, b := range buf[:n] {
		if b == 0 || (b < 32 && b != 9 && b != 10 && b != 13) { // Allow horizontal tab, line feed, carriage return
			return false
		}
	}

	return true
}

// isUnicodeBOM checks if the given bytes represent a Unicode BOM.
func isUnicodeBOM(bom []byte) bool {
	return (bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF) || // UTF-8
		(bom[0] == 0xFE && bom[1] == 0xFF) || // UTF-16 (BE)
		(bom[0] == 0xFF && bom[1] == 0xFE) || // UTF-16 (LE)
		(bom[0] == 0x00 && bom[1] == 0x00 && bom[2] == 0xFE && bom[3] == 0xFF) || // UTF-32 (BE)
		(bom[0] == 0xFF && bom[1] == 0xFE && bom[2] == 0x00 && bom[3] == 0x00) // UTF-32 (LE)
}

// readFileContent reads the content of a file, applying a size limit.
func readFileContent(filepath string, maxFileSize int64) (string, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return "", err
	}

	if fi.Size() > maxFileSize {
		return "", fmt.Errorf("file too large")
	}

	var builder strings.Builder
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		builder.WriteString(scanner.Text())
		builder.WriteString("\n")
	}

	if err := scanner.Err(); err != nil {
		return "", err
	}

	return builder.String(), nil
}

// appendToTree appends a directory or file to the tree structure string.
func appendToTree(builder *strings.Builder, relPath string, basePath string, isDir bool) {
	depth := strings.Count(relPath, string(os.PathSeparator))
	if depth == 0 {
		if isDir {
			builder.WriteString(relPath + string(os.PathSeparator) + "\n")
		} else {
			builder.WriteString(relPath + "\n")
		}
	} else {
		prefix := strings.Repeat("│   ", depth-1)
		// Check if it's the last element at the current depth
		isLast := isLastElement(basePath, relPath)
		if isLast {
			if isDir {
				builder.WriteString(prefix + "└── " + filepath.Base(relPath) + string(os.PathSeparator) + "\n")
			} else {
				builder.WriteString(prefix + "└── " + filepath.Base(relPath) + "\n")
			}
		} else {
			if isDir {
				builder.WriteString(prefix + "├── " + filepath.Base(relPath) + string(os.PathSeparator) + "\n")
			} else {
				builder.WriteString(prefix + "├── " + filepath.Base(relPath) + "\n")
			}
		}
	}
}

// isLastElement checks if the current element is the last at its depth.
func isLastElement(basePath, relPath string) bool {
	parts := strings.Split(relPath, string(os.PathSeparator))
	if len(parts) < 2 {
		return true // Root or single-level element is always last
	}

	parentDir := filepath.Join(basePath, strings.Join(parts[:len(parts)-1], string(os.PathSeparator)))
	currentBase := parts[len(parts)-1]

	entries, _ := os.ReadDir(parentDir)

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}

	return names[len(names)-1] == currentBase
}

// formatFilesContent creates a formatted string of file contents.
func formatFilesContent(files []FileInfo) string {
	var contentBuilder strings.Builder
	separator := strings.Repeat("=", 48) + "\n"

	// Find and prepend the README file, if it exists.
	for i, file := range files {
		if strings.ToLower(file.Path) == "readme.md" {
			contentBuilder.WriteString(separator)
			contentBuilder.WriteString(fmt.Sprintf("File: %s\n", file.Path))
			contentBuilder.WriteString(separator)
			contentBuilder.WriteString(file.Content + "\n\n")

			// Remove the README file from the slice to avoid duplicate processing.
			files = append(files[:i], files[i+1:]...)
			break
		}
	}

	// Process the rest of the files.
	for _, file := range files {
		contentBuilder.WriteString(separator)
		contentBuilder.WriteString(fmt.Sprintf("File: %s\n", file.Path))
		contentBuilder.WriteString(separator)
		contentBuilder.WriteString(file.Content + "\n\n")
	}

	return contentBuilder.String()
}

// countTokens estimates the number of tokens in the given text.
func countTokens(text string) (string, error) {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return "", err
	}

	totalTokens := len(enc.Encode(text, nil, nil))
	var formattedTokens string
	if totalTokens > 1000000 {
		formattedTokens = fmt.Sprintf("%.1fM", float64(totalTokens)/1000000)
	} else if totalTokens > 1000 {
		formattedTokens = fmt.Sprintf("%.1fk", float64(totalTokens)/1000)
	} else {
		formattedTokens = fmt.Sprintf("%d", totalTokens)
	}

	return formattedTokens, nil
}

// createSummary creates a summary of the repository analysis.
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

// parsePatterns parses a comma-separated string of patterns into a slice of strings.
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

// overrideIgnorePatterns overrides the default ignore patterns with include patterns.
func overrideIgnorePatterns(ignorePatterns []string, includePatterns []string) []string {
	var newIgnorePatterns []string
	for _, ignorePattern := range ignorePatterns {
		shouldIgnore := true
		for _, includePattern := range includePatterns {
			if matched, _ := filepath.Match(includePattern, ignorePattern); matched {
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

// parseIncludePatterns parses a pattern string into a slice of include patterns.
func parseIncludePatterns(pattern string) []string {
	// Remove leading and trailing spaces and split by comma
	parts := strings.Split(strings.TrimSpace(pattern), ",")

	var patterns []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Normalize the pattern: remove leading './' or '/'
		if strings.HasPrefix(part, "./") {
			part = part[2:]
		} else if strings.HasPrefix(part, "/") {
			part = part[1:]
		}

		// If the pattern is just a directory (ends with '/'), append '**' to match all subdirectories and files
		if strings.HasSuffix(part, "/") {
			patterns = append(patterns, part+"**")
		} else {
			patterns = append(patterns, part)
		}
	}

	return patterns
}

// ParseQuery parses the input query and returns the parsed information.
func ParseQuery(input string, maxFileSize int, fromWeb bool, includePatternsStr, excludePatternsStr string) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	if fromWeb {
		return nil, fmt.Errorf("web URL processing not implemented for CLI")
	}

	// Parse as local path
	absPath, err := filepath.Abs(input)
	if err != nil {
		return nil, fmt.Errorf("invalid local path: %v", err)
	}

	// Use the absolute path to construct the local_path
	result["local_path"] = absPath

	// Use the base of the absolute path as the slug
	result["slug"] = filepath.Base(absPath)
	result["subpath"] = "/"

	// Generate a unique ID using a part of the SHA-256 hash of the absolute path
	idBytes := sha256.Sum256([]byte(absPath))
	result["id"] = base58.Encode(idBytes[:16]) // Encode to Base58, using first 16 bytes for brevity

	// Set default ignore patterns and override with user-provided patterns
	defaultIgnorePatterns := []string{
		"*.o", "*.exe", "*.dll", "*.so", "*.dylib", // Object files, executables, and libraries
		"*.pyc", "*.pyo", "*.pyd", // Python
		"*.class", "*.jar", "*.war", "*.ear", // Java
		"*.exe", "*.dll", "*.obj", "*.o", "*.a", "*.lib", "*.so", // Windows
		"*.o", "*.a", "*.so", "*.dylib", // macOS
		"*.o", "*.a", "*.so", "*.so.*", // Linux
		"node_modules/", "vendor/", "bower_components/", // Node.js, PHP, Bower
		".git/", ".svn/", ".hg/", ".DS_Store/", // Version control and macOS system files
		".vscode/", ".idea/", ".sublime-project", ".sublime-workspace", ".idx", // IDE files
		"__pycache__/", "venv/", ".venv/", "env/", ".env/", // Python virtual environments and caches
		".idea/", ".vscode/", "*.sublime-project", "*.sublime-workspace", // IDE files
		"*.log", "*.tmp", "*.bak", // Temporary and log files
		".next/", ".nuxt/", "dist/", "build/", "target/", // Build and output directories
		"*.zip", "*.tar", "*.tar.gz", "*.rar", "*.7z", // Compressed files
		"*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.ico", // Image files
		"*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.flv", // Video files
		"*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg", // Audio files
		"*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx", // Document files
		".gitignore", ".DS_Store", // Other common files
		"*.pem", "*.cer", "*.crt", "*.key", // Certificate and key files
		"LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", // License files
		"README.MD", "README.md", "README.txt", "README.rst", "README.html", "README.pdf", // Readme files
	}

	if excludePatternsStr != "" {
		excludePatterns := parsePatterns(excludePatternsStr)
		result["ignore_patterns"] = append(defaultIgnorePatterns, excludePatterns...)
	} else {
		result["ignore_patterns"] = defaultIgnorePatterns
	}

	if includePatternsStr != "" {
		includePatterns := parseIncludePatterns(includePatternsStr)
		result["include_patterns"] = includePatterns
		result["ignore_patterns"] = overrideIgnorePatterns(result["ignore_patterns"].([]string), includePatterns)
	}

	result["max_file_size"] = int64(maxFileSize * 1024)

	return result, nil
}

// IngestFromQuery performs the main ingestion process based on the parsed query.
func IngestFromQuery(query map[string]interface{}) (string, string, string, error) {
	localPath := query["local_path"].(string)

	// Read and parse .gitignore files
	ignorePatterns := []string{}
	if _, err := os.Stat(filepath.Join(localPath, DefaultIgnoreFile)); err == nil {
		if patterns, err := parseGitIgnore(filepath.Join(localPath, DefaultIgnoreFile)); err == nil {
			ignorePatterns = append(ignorePatterns, patterns...)
		}
	}

	// Use custom ignore patterns if provided
	if patterns, ok := query["ignore_patterns"].([]string); ok {
		ignorePatterns = append(ignorePatterns, patterns...)
	}

	// Always ignore .git directory
	ignorePatterns = append(ignorePatterns, ".git/")

	maxFileSize := query["max_file_size"].(int64)
	files, tree, err := scanDirectory(localPath, ignorePatterns, maxFileSize)
	if err != nil {
		return "", "", "", fmt.Errorf("error scanning directory: %v", err)
	}

	filesContent := formatFilesContent(files)
	tokenCount, err := countTokens(tree + filesContent)
	if err != nil {
		return "", "", "", fmt.Errorf("error counting tokens: %v", err)
	}

	var repoName, branch, commit, subpath string
	if name, ok := query["slug"].(string); ok {
		repoName = name
	}
	if b, ok := query["branch"].(string); ok {
		branch = b
	}
	if c, ok := query["commit"].(string); ok {
		commit = c
	}
	if sp, ok := query["subpath"].(string); ok {
		subpath = sp
	}

	summary := createSummary(repoName, branch, commit, subpath, len(files), tokenCount)

	return summary, tree, filesContent, nil
}

func main() {
	inputPtr := flag.String("d", "", "Directory to process")
	outputPtr := flag.String("o", "", "Output file name")
	maxFileSizePtr := flag.Int("s", 10, "Maximum file size in KB")
	includePatternsPtr := flag.String("i", "", "Include patterns")
	excludePatternsPtr := flag.String("e", "", "Exclude patterns")

	flag.Parse()

	if *inputPtr == "" {
		log.Fatal("Error: input directory must be specified with -d")
	}

	if *outputPtr == "" {
		log.Fatal("Error: output file must be specified with -o")
	}

	query, err := ParseQuery(*inputPtr, *maxFileSizePtr, false, *includePatternsPtr, *excludePatternsPtr)
	if err != nil {
		log.Fatalf("Error parsing query: %v", err)
	}

	summary, tree, content, err := IngestFromQuery(query)
	if err != nil {
		log.Fatalf("Error during ingestion: %v", err)
	}

	// Write output to file
	outputFile := *outputPtr
	err = os.WriteFile(outputFile, []byte("Directory structure:\n"+tree+"\n"+content), 0644)
	if err != nil {
		log.Fatalf("Error writing output to file: %v", err)
	}

	fmt.Printf("Successfully processed directory: %s\n", *inputPtr)
	fmt.Printf("Output written to: %s\n", outputFile)
	fmt.Println(summary)
}

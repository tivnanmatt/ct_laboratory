import os

TOP_DIR = os.path.dirname(os.path.abspath(__file__))

def print_header(title):
    print("\n" + "=" * 80)
    print(f"= {title}")
    print("=" * 80 + "\n")

def print_file_contents(filepath):
    print_header(f"File: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        print(f.read())

def print_directory_structure(directory, prefix=""):
    entries = sorted(os.listdir(directory))
    for i, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{entry}")
        if os.path.isdir(path):
            print_directory_structure(path, prefix + ("    " if i == len(entries) - 1 else "│   "))

def main():
    print_header("Directory Structure")
    print(f"Root Directory: {TOP_DIR}")
    print_directory_structure(TOP_DIR)

    # Collect files
    files_to_print = []
    for root, _, files in os.walk(TOP_DIR):
        for file in files:
            if file in ["Makefile"] or file.endswith((".cpp", ".cu", ".py")):
                files_to_print.append(os.path.join(root, file))

    # Print contents of files
    for file in files_to_print:
        print_file_contents(file)

if __name__ == "__main__":
    main()

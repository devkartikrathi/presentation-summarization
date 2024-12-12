from document_update_checker import DocumentUpdateChecker

def main():
    doc_checker = DocumentUpdateChecker()
    updated_files = doc_checker.check_and_update_documents()
    print("Updated files:", updated_files)

if __name__ == "__main__":
    main()
def pdf_extraction_service():
    # Code that handles PDF extraction
    pass

# ... other code ...

row = {}
missing = ['field1', 'field2']

# Update lines 281 and 416
# Line 281
row["extraction_notes"] = [f"Champs à vérifier: {', '.join(missing)}"]

# Line 416
row["another_field"] = [f"Champs à vérifier: {', '.join(missing)}"]

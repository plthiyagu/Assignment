import pandas as pd

def chunk_text(text, chunk_size=1000, overlap=0):
    """
    Split text into chunks of chunk_size characters with optional overlap.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_chunked_dataframe(pdf, chunk_size=1000, overlap=200):
    chunked_docs = []
    for idx, row in pdf.iterrows():
        cik = row['cik']
        filename = row['filename']
        full_text = row['full_text'] if row['full_text'] else ""

        text_chunks = chunk_text(full_text, chunk_size, overlap)
        for i, chunk in enumerate(text_chunks):
            chunked_docs.append({
                "cik": cik,
                "filename": filename,
                "chunk_id": i,
                "chunk_text": chunk
            })

    df_chunks = pd.DataFrame(chunked_docs)
    print(f"Created {len(df_chunks)} text chunks from {len(pdf)} documents.")
    return df_chunks
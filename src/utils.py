import io
import pandas as pd

def universal_read_csv(uploaded_file_or_path):
    """
    Universally loads any CSV dataset or log file without crashing.
    Handles multiple encodings, delimiter variations, stream rewinding,
    and malformed rows.
    """
    encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'latin1', 'cp1252']
    delimiters = [',', ';', '\t', '|']
    is_buffer = hasattr(uploaded_file_or_path, 'seek')

    for encoding in encodings:
        for sep in delimiters:
            try:
                if is_buffer:
                    uploaded_file_or_path.seek(0)
                
                df = pd.read_csv(
                    uploaded_file_or_path,
                    encoding=encoding,
                    sep=sep,
                    on_bad_lines='skip',
                    engine='c' if sep == ',' else 'python'
                )
                
                if df.shape[1] >= 1:
                    df.columns = [str(col).strip() for col in df.columns]
                    return df
            except Exception:
                continue

    # Fallback attempt using Python engine auto-delimiter detection
    try:
        if is_buffer:
            uploaded_file_or_path.seek(0)
        df = pd.read_csv(
            uploaded_file_or_path,
            encoding='iso-8859-1',
            sep=None,
            engine='python',
            on_bad_lines='skip'
        )
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Could not parse file. Details: {e}")
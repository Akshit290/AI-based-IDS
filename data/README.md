# Network Traffic Data

Place your network traffic datasets here in CSV format.

## Supported Formats

### CSV Format
- Headers: feature1, feature2, ..., attack (target column)
- Rows: One record per line
- Separator: Comma

### Expected Columns
- duration (float): Connection duration in seconds
- protocol_type (string): TCP, UDP, ICMP
- service (string): HTTP, FTP, DNS, SSH, SMTP, etc.
- src_bytes (float): Bytes sent from source
- dst_bytes (float): Bytes sent to destination
- flag (string): Connection status flags
- land (int): 1 if same source/destination
- wrong_fragment (int): Number of wrong fragments
- urgent (int): Number of urgent packets
- hot (int): Number of hot indicators
- attack (int): Target label (0=normal, 1=intrusion)

## Sample Data

Run the data generator to create sample data:
```bash
python src/data_Pipelines/generate_sample_data.py
```

This will create `network_traffic.csv` with 10,000 sample records.

## Real Datasets

- **NSL-KDD**: https://www.ics.uci.edu/~mlearn/databases/nsl-kdd/
- **CICIDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **CICIDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Usage

```python
from src.data_Pipelines.data_pipeline import DataPipeline

pipeline = DataPipeline()
X_train, X_test, y_train, y_test = pipeline.prepare_data('data/your_data.csv')
```

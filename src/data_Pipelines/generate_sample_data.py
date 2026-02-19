"""
Script to generate sample network traffic data for testing and training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_samples=10000, output_file='data/network_traffic.csv'):
    """
    Generate sample network traffic dataset.
    
    Features:
    - duration: Length of connection in seconds
    - protocol_type: TCP, UDP, ICMP
    - service: HTTP, FTP, DNS, etc.
    - src_bytes: Bytes sent from source
    - dst_bytes: Bytes sent to destination
    - flag: Connection status flags
    - land: 1 if connection between same host/port
    - wrong_fragment: Number of wrong fragments
    - urgent: Number of urgent packets
    - hot: Number of "hot" indicators
    - num_failed_logins: Failed login attempts
    - num_compromised: Compromised condition
    - root_shell: 1 if root shell obtained
    - su_attempted: 1 if su root command attempted
    - num_root: Number of root accesses
    - num_file_creations: Number of file creation operations
    - num_shells: Number of shell prompts
    - num_access_files: Number of operations on access control files
    - num_outbound_cmds: Number of outbound commands
    - is_host_login: 1 if login belongs to "hot" list
    - is_guest_login: 1 if login is guest
    - attack: 1 if attack, 0 if normal
    """
    
    np.random.seed(42)
    
    print(f"Generating {n_samples} sample network traffic records...")
    
    # Generate base features
    data = {
        'duration': np.random.exponential(100, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'dns', 'ssh', 'smtp'], n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(1000, n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1'], n_samples),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.poisson(0.5, n_samples),
        'urgent': np.random.poisson(0.2, n_samples),
        'hot': np.random.poisson(0.1, n_samples),
        'num_failed_logins': np.random.poisson(0.5, n_samples),
        'num_compromised': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'root_shell': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'su_attempted': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'num_root': np.random.poisson(0.1, n_samples),
        'num_file_creations': np.random.poisson(1, n_samples),
        'num_shells': np.random.poisson(0.5, n_samples),
        'num_access_files': np.random.poisson(0.2, n_samples),
        'num_outbound_cmds': np.random.poisson(0.1, n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate attack labels (20% attacks, 80% normal)
    attack_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    df['attack'] = 0
    df.loc[attack_indices, 'attack'] = 1
    
    # Modify attack records to have different characteristics
    for idx in attack_indices:
        attack_type = np.random.choice(['dos', 'probe', 'r2l', 'u2r'])
        
        if attack_type == 'dos':
            # DDoS characteristics: high volume
            df.at[idx, 'src_bytes'] = np.random.exponential(10000)
            df.at[idx, 'dst_bytes'] = np.random.exponential(10000)
            df.at[idx, 'duration'] = np.random.exponential(500)
        elif attack_type == 'probe':
            # Port scanning: many failed connections
            df.at[idx, 'wrong_fragment'] = np.random.poisson(5)
            df.at[idx, 'urgent'] = np.random.poisson(2)
        elif attack_type == 'r2l':
            # Remote to local: failed logins
            df.at[idx, 'num_failed_logins'] = np.random.poisson(10)
            df.at[idx, 'duration'] = np.random.exponential(1000)
        else:  # u2r
            # User to root: privilege escalation
            df.at[idx, 'su_attempted'] = 1
            df.at[idx, 'root_shell'] = 1
            df.at[idx, 'num_root'] = np.random.poisson(5)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[OK] Sample data saved to {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Normal samples: {(df['attack'] == 0).sum()}")
    print(f"  Attack samples: {(df['attack'] == 1).sum()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


if __name__ == '__main__':
    generate_sample_data(n_samples=10000, output_file='data/network_traffic.csv')

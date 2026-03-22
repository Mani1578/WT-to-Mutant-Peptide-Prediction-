# WT-to-Mutant-Peptide-Prediction-
We developed a novel hybrid deep learning framework that integrates ESM-2 embeddings, a Bi_LSTM with attention, and numeric biochemical and microenvironmental descriptors to assess the functional impact of p53 variants and classify p53 peptide sequences into wild-type-like or mutant-like categories. 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset and handle potential dtype warnings
df = pd.read_csv("p53_cdna_to_peptides.csv", dtype={
    'WAF1_': str, 'MDM2_': str, 'BAX_': str, '__14_3_3_s': str, 'AIP_': str,
    'GADD45_': str, 'NOXA_': str, 'p53R2_': str, 'Pathogenicity': str
})


# Adjust column names if needed
sequences = df['Protein_Variant'].astype(str).values
# Convert WT_Peptide to numerical labels: 0 for 'Mutant_Peptide': 1 
labels = df['WT_Peptide'].apply(lambda x: 1 if 'WT_Peptide' in x.lower() else 0).values


# 2. Encode amino acids
aa_vocab = sorted(set("".join(sequences)))
aa_to_int = {aa: i+1 for i, aa in enumerate(aa_vocab)}  # reserve 0 for padding
int_sequences = [[aa_to_int[aa] for aa in seq] for seq in sequences]

# 3. Pad sequences
maxlen = max(len(seq) for seq in int_sequences)
X = pad_sequences(int_sequences, maxlen=maxlen, padding='post')
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Build deep learning model (you can switch LSTM <-> GRU)
def build_model(rnn_type="LSTM"):
    model = Sequential()
    model.add(Embedding(input_dim=len(aa_to_int)+1, output_dim=64, input_length=maxlen))

    if cnn_type == "LSTM":
        model.add(Bidirectional(LSTM(128, return_sequences=False)))
    else:
        model.add(Bidirectional(GRU(128, return_sequences=False)))

    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train LSTM model
model = build_model("LSTM")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80, batch_size=32, verbose=1
)

# 5. Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Plot training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.legend()
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset and handle potential dtype warnings
df = pd.read_csv("p53_cdna_to_peptides.csv", dtype={
    'WAF1_': str, 'MDM2_': str, 'BAX_': str, '__14_3_3_s': str, 'AIP_': str,
    'GADD45_': str, 'NOXA_': str, 'p53R2_': str, 'Pathogenicity': str
})

# Adjust column names if needed
sequences = df['Protein_Variant'].astype(str).values
# Convert WT_Peptide to numerical labels: 0 for 'Mutant_Peptide': 1 
labels = df['WT_Peptide'].apply(lambda x: 1 if 'WT_Peptide' in x.lower() else 0).values

# 2. Encode amino acids
aa_vocab = sorted(set("".join(sequences)))
aa_to_int = {aa: i+1 for i, aa in enumerate(aa_vocab)}  # reserve 0 for padding
int_sequences = [[aa_to_int[aa] for aa in seq] for seq in sequences]

# 3. Pad sequences
maxlen = max(len(seq) for seq in int_sequences)
X = pad_sequences(int_sequences, maxlen=maxlen, padding='post')
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Build Hybrid CNN + BiLSTM model
def build_cnn_bilstm():
    model = Sequential()
    model.add(Embedding(input_dim=len(aa_to_int)+1, output_dim=128, input_length=maxlen))

    # Convolutional feature extractor (detect motifs)
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Sequential feature learner
    model.add(Bidirectional(LSTM(128, return_sequences=False)))

    # Fully connected layers
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train CNN+BiLSTM model
model = build_cnn_bilstm()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80, batch_size=32, verbose=1
)

# 5. Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Plot training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve (CNN+BiLSTM)")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve (CNN+BiLSTM)")
plt.legend()
plt.show()

--------------------------------------------------------------------------------------------------------------------------------------------------



#!/usr/bin/env python3
"""
Hybrid ACP predictor for p53 peptides (PSSM optional).
- Uses sequence branch (Embedding -> CNN + BiLSTM)
- Uses AAC (20D) + physicochemical aggregated descriptors (3D)
- Trains a binary classifier (ACP / non-ACP) with PyTorch
Requirements: pandas, numpy, scikit-learn, torch
Run: python acp_hybrid_pipeline.py
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- USER PARAMETERS ----------
CSV_PATH = "p53_cdna_to_peptides.csv"   # change if necessary
OUTPUT_DIR = "acp_hybrid_output"
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
# -------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- helper amino acid maps ----------
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
aa2idx = {aa: i for i, aa in enumerate(AA_LIST)}
PAD_IDX = len(AA_LIST)  # padding index

# Kyte-Doolittle hydrophobicity, simple charge and polarity mapping
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
         'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'A':0,'R':1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,'G':0,'H':1,'I':0,'L':0,'K':1,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
POLAR = {'A':0,'R':1,'N':1,'D':1,'C':0,'Q':1,'E':1,'G':0,'H':1,'I':0,'L':0,'K':1,'M':0,'F':0,'P':0,'S':1,'T':1,'W':0,'Y':1,'V':0}

# ---------- data loading & column selection ----------
def load_and_select(csv_path):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    # candidate peptide columns (common names)
    peptide_candidates = [c for c in cols if any(k in c.lower() for k in ["mutant_peptide","mutant","mut_peptide","mutant_pep","mut_pep","wt_peptide","wt_peptide","wt_pep","peptide","pept","sequence"])]
    label_candidates = [c for c in cols if any(k in c.lower() for k in ["label","is_acp","target","type","class","prediction_label","pathogenicity"])]

    # prefer Mutant_Peptide, then Mutant_Peptide-like, then WT_Peptide
    seq_col = None
    if "Mutant_Peptide" in df.columns:
        seq_col = "Mutant_Peptide"
    elif "Mutant_Peptide".lower() in [c.lower() for c in df.columns]:
        # find exact case-insensitive match
        seq_col = [c for c in df.columns if c.lower()=="mutant_peptide"][0]
    else:
        # fallbacks
        for c in peptide_candidates:
            if "mutant" in c.lower():
                seq_col = c; break
        if seq_col is None and len(peptide_candidates)>0:
            # prefer explicit WT/Mutant peptide columns
            seq_col = peptide_candidates[0]

    if seq_col is None:
        raise ValueError("Could not find peptide column automatically. Please provide column name.")

    # choose label column
    if "Prediction_Label" in df.columns:
        label_col = "Prediction_Label"
    elif len(label_candidates)>0:
        label_col = label_candidates[0]
    else:
        # default to 'Type' or second column
        label_col = "Type" if "Type" in df.columns else df.columns[1]

    return df, seq_col, label_col

def clean_seq(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    return "".join([c for c in s if c in AA_LIST])

# ---------- feature functions ----------
def aac(seq):
    vec = np.zeros(len(AA_LIST), dtype=float)
    for c in seq:
        if c in aa2idx:
            vec[aa2idx[c]] += 1
    if len(seq) > 0:
        vec = vec / len(seq)
    return vec

def phys_agg(seq):
    if len(seq)==0:
        return np.array([0.0,0.0,0.0])
    charges = [CHARGE[c] for c in seq if c in CHARGE]
    polys = [POLAR[c] for c in seq if c in POLAR]
    hydros = [HYDRO[c] for c in seq if c in HYDRO]
    return np.array([np.mean(charges), np.mean(polys), np.mean(hydros)])

def seq_to_idx(seq, max_len):
    idxs = [aa2idx[c] for c in seq if c in aa2idx]
    if len(idxs) < max_len:
        idxs = idxs + [PAD_IDX] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return np.array(idxs, dtype=int)

# ---------- PyTorch model ----------
class HybridACPModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len, aac_dim=20, phys_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.conv1 = nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(emb_dim, 64, batch_first=True, bidirectional=True)
        self.aac_dense = nn.Sequential(nn.Linear(aac_dim, 64), nn.ReLU())
        self.phys_dense = nn.Sequential(nn.Linear(phys_dim, 16), nn.ReLU())
        fused_dim = 128 + 128 + 64 + 16
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, seq_idx, aac_feat, phys_feat):
        emb = self.embedding(seq_idx)  # (batch, L, emb_dim)
        x = emb.permute(0,2,1)         # (batch, emb_dim, L)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)   # (batch, 128)
        lstm_out, _ = self.lstm(emb)   # (batch, L, 128)
        lstm_feat = lstm_out[:, -1, :] # (batch, 128)
        aac_out = self.aac_dense(aac_feat)
        phys_out = self.phys_dense(phys_feat)
        concat = torch.cat([x, lstm_feat, aac_out, phys_out], dim=1)
        out = self.fc(concat)
        return out

# ---------- main pipeline ----------
def main():
    print("Loading CSV...")
    df, seq_col, label_col = load_and_select(CSV_PATH)
    print("Using sequence column:", seq_col, "and label column:", label_col)
    df['sequence'] = df[seq_col].astype(str).apply(clean_seq)
    # Coerce label to 0/1
    if df[label_col].dtype == object:
        df['label'] = df[label_col].map(lambda v: 1 if str(v).strip().lower() in ["1","yes","true","acp","pos","positive"] else 0)
    else:
        df['label'] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

    df = df[df['sequence'].str.len() > 0].reset_index(drop=True)
    print("Total sequences:", len(df))

    sequences = df['sequence'].tolist()
    labels = df['label'].values

    max_len = max(len(s) for s in sequences)
    print("Max peptide length:", max_len)

    # Build features
    X_aac = np.array([aac(s) for s in sequences])
    X_phys = np.array([phys_agg(s) for s in sequences])
    X_seq_idx = np.array([seq_to_idx(s, max_len) for s in sequences])

    # Train-test
    stratify = labels if len(np.unique(labels)) > 1 else None
    X_seq_train, X_seq_test, X_aac_train, X_aac_test, X_phys_train, X_phys_test, y_train, y_test = train_test_split(
        X_seq_idx, X_aac, X_phys, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Scale AAC and phys
    aac_scaler = StandardScaler()
    phys_scaler = StandardScaler()
    X_aac_train = aac_scaler.fit_transform(X_aac_train)
    X_aac_test = aac_scaler.transform(X_aac_test)
    X_phys_train = phys_scaler.fit_transform(X_phys_train)
    X_phys_test = phys_scaler.transform(X_phys_test)

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_seq_train_t = torch.tensor(X_seq_train, dtype=torch.long, device=device)
    X_seq_test_t = torch.tensor(X_seq_test, dtype=torch.long, device=device)
    X_aac_train_t = torch.tensor(X_aac_train, dtype=torch.float32, device=device)
    X_aac_test_t = torch.tensor(X_aac_test, dtype=torch.float32, device=device)
    X_phys_train_t = torch.tensor(X_phys_train, dtype=torch.float32, device=device)
    X_phys_test_t = torch.tensor(X_phys_test, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

    print("Device:", device, "Train size:", X_seq_train_t.shape[0], "Test size:", X_seq_test_t.shape[0])

    # Model init
    vocab_size = len(AA_LIST) + 1  # +1 for padding
    model = HybridACPModel(vocab_size=vocab_size, emb_dim=32, max_len=max_len).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    n_train = X_seq_train_t.shape[0]
    for epoch in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            seq_batch = X_seq_train_t[idx]
            aac_batch = X_aac_train_t[idx]
            phys_batch = X_phys_train_t[idx]
            y_batch = y_train_t[idx]
            optimizer.zero_grad()
            preds = model(seq_batch, aac_batch, phys_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * seq_batch.size(0)
        epoch_loss /= n_train

        # eval
        model.eval()
        with torch.no_grad():
            preds_test = model(X_seq_test_t, X_aac_test_t, X_phys_test_t).cpu().numpy().squeeze()
            pred_labels = (preds_test >= 0.5).astype(int)
            acc = accuracy_score(y_test, pred_labels)
            try:
                auc = roc_auc_score(y_test, preds_test)
            except:
                auc = float('nan')
        print(f"Epoch {epoch}/{EPOCHS} - Loss {epoch_loss:.4f} - Test Acc {acc:.4f} - ROC AUC {auc:.4f}")

    # Save model and outputs
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "hybrid_acp_model.pt"))
    out_df = pd.DataFrame({
        "sequence": df['sequence'].iloc[len(df)-len(preds_test):].values[:len(preds_test)],
        "true_label": y_test,
        "pred_score": preds_test,
        "pred_label": (preds_test>=0.5).astype(int)
    })
    out_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print("Model and predictions saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()

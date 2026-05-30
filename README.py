import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

# ============================================================
# STEP 1: Load the p53 reference FASTA
# ============================================================
def load_p53_fasta(fasta_path):
    """Load the wild-type p53 protein sequence from FASTA file."""
    record = next(SeqIO.parse(fasta_path, "fasta"))
    wt_sequence = str(record.seq)
    print(f"Loaded p53 sequence: {record.id}")
    print(f"WT sequence length : {len(wt_sequence)} amino acids")
    print(f"First 20 aa        : {wt_sequence[:20]}")
    return wt_sequence

# ============================================================
# STEP 2: Load the mutation dataset
# ============================================================
def load_mutations(csv_path):
    """Load and clean the p53 mutation dataset."""
    df = pd.read_csv(csv_path, dtype={
        'WAF1_': str, 'MDM2_': str, 'BAX_': str, '__14_3_3_s': str,
        'AIP_': str, 'GADD45_': str, 'NOXA_': str, 'p53R2_': str,
        'Pathogenicity': str
    })

    # Keep only missense / base substitution mutations for sequence generation
    df = df[df['Mutation_Type'] == 'B'].copy()          # B = base substitution
    df = df[df['WT_AA_1'].notna()].copy()
    df = df[df['Mutant_AA_1'].notna()].copy()
    df = df[df['Codon'].notna()].copy()
    df['Codon'] = df['Codon'].astype(int)

    print(f"\nTotal mutations loaded  : {len(df)}")
    print(f"Unique codon positions  : {df['Codon'].nunique()}")
    print(f"Sample mutations:\n{df[['Codon','WT_AA_1','Mutant_AA_1','Wt_Codon','Mutant_Codon']].head(5)}")
    return df

# ============================================================
# STEP 3: Generate WT and Mutant sequences
# ============================================================
def generate_sequences(wt_full_sequence, df):
    """
    For each mutation in the dataset:
    - Extract a window around the mutation site (WT peptide)
    - Apply the amino acid substitution (Mutant peptide)

    Returns a DataFrame with WT_Sequence and Mutant_Sequence columns.
    """
    records = []
    window = 10  # ±10 amino acids around mutation site

    for _, row in df.iterrows():
        pos = int(row['Codon']) - 1           # convert to 0-indexed
        wt_aa  = str(row['WT_AA_1']).strip()
        mut_aa = str(row['Mutant_AA_1']).strip()

        # Skip if position is out of range
        if pos < 0 or pos >= len(wt_full_sequence):
            continue

        # Verify WT amino acid matches the reference
        ref_aa = wt_full_sequence[pos]
        if ref_aa != wt_aa:
            continue                          # mismatch — skip this entry

        # --- Full-length sequences ---
        wt_full  = wt_full_sequence
        mut_full = wt_full_sequence[:pos] + mut_aa + wt_full_sequence[pos+1:]

        # --- Windowed peptide sequences (local context) ---
        start = max(0, pos - window)
        end   = min(len(wt_full_sequence), pos + window + 1)

        wt_peptide  = wt_full_sequence[start:end]
        mut_peptide = wt_full_sequence[start:pos] + mut_aa + wt_full_sequence[pos+1:end]

        records.append({
            'Database_ID'       : row.get('Database_ID', ''),
            'Codon_Position'    : pos + 1,            # 1-indexed
            'WT_AA'             : wt_aa,
            'Mutant_AA'         : mut_aa,
            'Wt_Codon'          : row.get('Wt_Codon', ''),
            'Mutant_Codon'      : row.get('Mutant_Codon', ''),
            'Domain'            : row.get('Domain', ''),
            'Pathogenicity'     : row.get('Pathogenicity', ''),
            'Mutation_Type'     : row.get('Type', ''),        # Ts/Tv/Td

            # Full-length sequences
            'WT_Full_Sequence'  : wt_full,
            'Mutant_Full_Sequence': mut_full,

            # Windowed peptide sequences (±10 aa)
            'WT_Peptide'        : wt_peptide,
            'Mutant_Peptide'    : mut_peptide,

            # Label for ML
            'Label'             : 0 if row.get('Pathogenicity','').strip() == 'Benign'
                                   else 1   # 1 = Pathogenic/Likely Pathogenic/VUS
        })

    result_df = pd.DataFrame(records)
    return result_df

# ============================================================
# STEP 4: Save outputs
# ============================================================
def save_outputs(result_df, out_csv="p53_wt_mutant_sequences.csv",
                             out_fasta_wt="p53_wt_peptides.fasta",
                             out_fasta_mut="p53_mutant_peptides.fasta"):

    # Save to CSV
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV → {out_csv}  ({len(result_df)} records)")

    # Save WT peptides as FASTA
    with open(out_fasta_wt, 'w') as f:
        for _, row in result_df.iterrows():
            f.write(f">{row['Database_ID']}_pos{row['Codon_Position']}_{row['WT_AA']}_WT\n")
            f.write(f"{row['WT_Peptide']}\n")
    print(f"Saved WT FASTA  → {out_fasta_wt}")

    # Save Mutant peptides as FASTA
    with open(out_fasta_mut, 'w') as f:
        for _, row in result_df.iterrows():
            f.write(f">{row['Database_ID']}_pos{row['Codon_Position']}_{row['WT_AA']}>{row['Mutant_AA']}_MUT\n")
            f.write(f"{row['Mutant_Peptide']}\n")
    print(f"Saved Mutant FASTA → {out_fasta_mut}")

    return out_csv

# ============================================================
# STEP 5: Summary statistics
# ============================================================
def print_summary(result_df):
    print("\n" + "="*50)
    print("SEQUENCE GENERATION SUMMARY")
    print("="*50)
    print(f"Total sequences generated  : {len(result_df)}")
    print(f"Unique codon positions      : {result_df['Codon_Position'].nunique()}")
    print(f"WT peptide length (avg)     : {result_df['WT_Peptide'].str.len().mean():.1f} aa")
    print(f"Mutant peptide length (avg) : {result_df['Mutant_Peptide'].str.len().mean():.1f} aa")
    print(f"\nPathogenicity distribution:")
    print(result_df['Pathogenicity'].value_counts())
    print(f"\nMutation type distribution:")
    print(result_df['Mutation_Type'].value_counts())
    print(f"\nTop 10 mutated domains:")
    print(result_df['Domain'].value_counts().head(10))
    print(f"\nSample output:")
    print(result_df[['Codon_Position','WT_AA','Mutant_AA',
                      'WT_Peptide','Mutant_Peptide','Pathogenicity']].head(5).to_string())

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # --- Paths ---
    FASTA_PATH = "p53.fasta"                          # your p53 FASTA file
    CSV_PATH   = "final dataset P53.csv"              # your mutation dataset

    # --- Run pipeline ---
    wt_sequence = load_p53_fasta(FASTA_PATH)
    df_mutations = load_mutations(CSV_PATH)
    result_df    = generate_sequences(wt_sequence, df_mutations)
    save_outputs(result_df)
    print_summary(result_df)

-----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import EsmModel, EsmTokenizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — tuned for small imbalanced dataset
# ============================================================
CONFIG = {
    'csv_path'        : 'p53_cdna_to_peptides.csv',
    'esm_model'       : 'facebook/esm2_t6_8M_UR50D',
    'max_len'         : 21,
    'embed_dim'       : 320,

    # Model
    'cnn_filters'     : 128,        # reduced to avoid overfit on small data
    'cnn_kernel'      : 3,
    'lstm_hidden'     : 256,        # reduced
    'num_lstm_layers' : 2,
    'dropout'         : 0.5,        # increased

    # Training
    'batch_size'      : 16,         # smaller for better gradient signal
    'epochs'          : 80,
    'lr'              : 1e-4,
    'weight_decay'    : 1e-4,
    'label_smoothing' : 0.1,        # prevents overconfident predictions
    'focal_gamma'     : 2.0,        # focal loss for class imbalance
    'k_folds'         : 5,          # cross-validation
    'test_size'       : 0.20,
    'random_state'    : 42,
    'device'          : 'cuda' if torch.cuda.is_available() else 'cpu',

    # Augmentation
    'augment'         : True,
    'aug_prob'        : 0.3,        # 30% chance per sequence
}
print(f"Device       : {CONFIG['device']}")
print(f"K-Folds      : {CONFIG['k_folds']}")
print(f"Label smooth : {CONFIG['label_smoothing']}")

# ============================================================
# STEP 1: Load Dataset
# ============================================================
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, dtype={
        'WAF1_': str, 'MDM2_': str, 'BAX_': str, '__14_3_3_s': str,
        'AIP_': str, 'GADD45_': str, 'NOXA_': str, 'p53R2_': str,
        'Pathogenicity': str
    })
    pep_df = df[df['WT_Peptide'].notna() & df['Mutant_Peptide'].notna()].copy()

    wt = pep_df[['WT_Peptide', 'Pathogenicity']].rename(
        columns={'WT_Peptide': 'sequence'})
    wt['label'] = 0

    mut = pep_df[['Mutant_Peptide', 'Pathogenicity']].rename(
        columns={'Mutant_Peptide': 'sequence'})
    mut['label'] = 1

    combined = pd.concat([wt, mut], ignore_index=True)
    combined  = combined.drop_duplicates(subset=['sequence', 'label'])
    combined  = combined.dropna(subset=['sequence'])

    print(f"Total unique samples : {len(combined)}")
    print(f"WT  (label=0)        : {(combined['label']==0).sum()}")
    print(f"Mutant (label=1)     : {(combined['label']==1).sum()}")
    print(f"Imbalance ratio      : 1:{(combined['label']==1).sum()/(combined['label']==0).sum():.1f}")
    return combined

# ============================================================
# STEP 2: Sequence Augmentation (for small dataset)
# ============================================================
AA_SIMILAR = {
    # Conservative substitutions — similar physicochemical properties
    'A': ['G','S'],    'G': ['A','S'],    'S': ['T','A'],    'T': ['S','V'],
    'V': ['I','L'],    'I': ['L','V'],    'L': ['I','V'],    'F': ['Y','W'],
    'Y': ['F','W'],    'W': ['F','Y'],    'D': ['E','N'],    'E': ['D','Q'],
    'N': ['D','Q'],    'Q': ['E','N'],    'K': ['R','H'],    'R': ['K','H'],
    'H': ['K','R'],    'C': ['S','A'],    'M': ['L','I'],    'P': ['A','G'],
}

def augment_sequence(seq, prob=0.3):
    """
    Conservative amino acid substitution at NON-center positions.
    Center position (mutation site) is preserved.
    """
    seq_list   = list(seq)
    center_idx = len(seq) // 2

    for i, aa in enumerate(seq_list):
        if i == center_idx:
            continue                           # never touch mutation site
        if np.random.random() < prob and aa in AA_SIMILAR:
            seq_list[i] = np.random.choice(AA_SIMILAR[aa])

    return ''.join(seq_list)

def create_augmented_dataset(df, aug_factor=3, aug_prob=0.3):
    """
    Over-sample minority class (WT) with augmentation.
    Apply light augmentation to majority class too.
    """
    wt_df  = df[df['label'] == 0].copy()
    mut_df = df[df['label'] == 1].copy()

    # Augment WT (minority) more aggressively
    augmented_wt = []
    for _ in range(aug_factor):
        temp = wt_df.copy()
        temp['sequence'] = temp['sequence'].apply(
            lambda s: augment_sequence(s, aug_prob))
        augmented_wt.append(temp)

    # Light augmentation on mutant (majority)
    aug_mut = mut_df.copy()
    aug_mut['sequence'] = aug_mut['sequence'].apply(
        lambda s: augment_sequence(s, aug_prob * 0.5))

    combined_aug = pd.concat(
        [df, *augmented_wt, aug_mut], ignore_index=True)

    print(f"\nAfter augmentation:")
    print(f"  WT     : {(combined_aug['label']==0).sum()}")
    print(f"  Mutant : {(combined_aug['label']==1).sum()}")
    return combined_aug

# ============================================================
# STEP 3: ESM-2 Embeddings
# ============================================================
def get_esm2_embeddings(sequences, tokenizer, esm_model,
                         device, batch_size=64, max_len=21):
    esm_model.eval()
    esm_model.to(device)
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i: i + batch_size]
        spaced = [' '.join(list(str(s))) for s in batch_seqs]

        tokens = tokenizer(
            spaced,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_len + 2
        ).to(device)

        with torch.no_grad():
            outputs = esm_model(**tokens)

        hidden = outputs.last_hidden_state[:, 1:-1, :]  # remove CLS/EOS

        B, L, D = hidden.shape
        if L < max_len:
            pad    = torch.zeros(B, max_len - L, D).to(device)
            hidden = torch.cat([hidden, pad], dim=1)
        else:
            hidden = hidden[:, :max_len, :]

        all_embeddings.append(hidden.cpu())

        if (i // batch_size + 1) % 20 == 0:
            print(f"  Embedded {min(i+batch_size, len(sequences))}"
                  f"/{len(sequences)}...")

    return torch.cat(all_embeddings, dim=0)   # (N, max_len, embed_dim)

# ============================================================
# STEP 4: Focal Loss (handles class imbalance better than CE)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    gamma=2 is standard; higher gamma = more focus on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Label smoothing
        n_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), 1)
        smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                         self.label_smoothing / n_classes

        log_prob = F.log_softmax(inputs, dim=1)
        prob     = torch.exp(log_prob)

        # Focal weight
        focal_weight = (1 - prob) ** self.gamma
        loss = -(smooth_targets * focal_weight * log_prob).sum(dim=1)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss    = alpha_t * loss

        return loss.mean()

# ============================================================
# STEP 5: Dataset Class with Mixup Augmentation
# ============================================================
class PeptideDataset(Dataset):
    def __init__(self, embeddings, labels, augment=False):
        self.embeddings = embeddings
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.augment    = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        lbl = self.labels[idx]

        # Runtime Gaussian noise augmentation (only during training)
        if self.augment:
            noise = torch.randn_like(emb) * 0.01
            emb   = emb + noise

        return emb, lbl

# ============================================================
# STEP 6: Improved CNN + Bi-LSTM + ESM-2 
# ============================================================
class ImprovedCNN_BiLSTM_ESM2(nn.Module):
    """
    Improvements over baseline:
    1. Multi-scale CNN (kernel 3, 5, 7) captures patterns at different lengths
    2. Residual connections prevent vanishing gradients
    3. Multi-head attention (4 heads) instead of single attention
    4. Layer normalization for stable training
    5. Deeper FC with residual skip connection
    """
    def __init__(self, embed_dim=320, cnn_filters=128,
                 lstm_hidden=128, num_lstm_layers=2,
                 dropout=0.5, num_classes=2):
        super(ImprovedCNN_BiLSTM_ESM2, self).__init__()

        self.embed_dim = embed_dim

        # --- Input projection ---
        self.input_proj = nn.Linear(embed_dim, cnn_filters)
        self.input_norm = nn.LayerNorm(cnn_filters)

        # --- Multi-scale CNN (3 kernel sizes) ---
        self.conv_k3 = nn.Conv1d(cnn_filters, cnn_filters,
                                  kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(cnn_filters, cnn_filters,
                                  kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(cnn_filters, cnn_filters,
                                  kernel_size=7, padding=3)

        self.bn_k3 = nn.BatchNorm1d(cnn_filters)
        self.bn_k5 = nn.BatchNorm1d(cnn_filters)
        self.bn_k7 = nn.BatchNorm1d(cnn_filters)

        # Merge multi-scale: 3 * cnn_filters → cnn_filters
        self.cnn_merge = nn.Conv1d(cnn_filters * 3, cnn_filters,
                                    kernel_size=1)
        self.cnn_merge_bn = nn.BatchNorm1d(cnn_filters)

        # --- Residual CNN Block 2 ---
        self.conv2     = nn.Conv1d(cnn_filters, cnn_filters,
                                    kernel_size=3, padding=1)
        self.bn2       = nn.BatchNorm1d(cnn_filters)

        # --- Bidirectional LSTM ---
        self.bilstm = nn.LSTM(
            input_size  = cnn_filters,
            hidden_size = lstm_hidden,
            num_layers  = num_lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if num_lstm_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)

        # --- Multi-Head Attention (4 heads) ---
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim   = lstm_hidden * 2,
            num_heads   = 4,
            dropout     = dropout,
            batch_first = True
        )
        self.attn_norm = nn.LayerNorm(lstm_hidden * 2)

        # --- Fully Connected with skip connection ---
        fc_in = lstm_hidden * 2
        self.fc1      = nn.Linear(fc_in, 256)
        self.fc2      = nn.Linear(256, 128)
        self.fc3      = nn.Linear(128, num_classes)
        self.fc_skip  = nn.Linear(fc_in, 128)   # skip connection

        self.fc1_norm = nn.LayerNorm(256)
        self.fc2_norm = nn.LayerNorm(128)

        self.dropout  = nn.Dropout(dropout)
        self.relu     = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len=21, embed_dim=320)
        B, L, D = x.shape

        # --- Input projection ---
        x = self.input_proj(x)          # (B, L, cnn_filters)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)         # (B, cnn_filters, L) for CNN

        # --- Multi-scale CNN ---
        f3 = F.relu(self.bn_k3(self.conv_k3(x)))   # kernel 3
        f5 = F.relu(self.bn_k5(self.conv_k5(x)))   # kernel 5
        f7 = F.relu(self.bn_k7(self.conv_k7(x)))   # kernel 7

        multi = torch.cat([f3, f5, f7], dim=1)      # (B, 3*filters, L)
        x     = F.relu(self.cnn_merge_bn(
            self.cnn_merge(multi)))                  # (B, filters, L)

        # --- Residual CNN Block ---
        residual = x
        x        = F.relu(self.bn2(self.conv2(x)))
        x        = x + residual                      # residual connection
        x        = self.dropout(x)

        # --- Bi-LSTM ---
        x = x.permute(0, 2, 1)                      # (B, L, filters)
        lstm_out, _ = self.bilstm(x)                 # (B, L, lstm_hidden*2)
        lstm_out    = self.lstm_norm(lstm_out)

        # --- Multi-Head Self-Attention ---
        attn_out, _ = self.multihead_attn(
            lstm_out, lstm_out, lstm_out)            # (B, L, lstm_hidden*2)
        x = self.attn_norm(lstm_out + attn_out)      # residual + norm

        # Global average + max pooling then concatenate
        avg_pool = x.mean(dim=1)                     # (B, lstm_hidden*2)
        max_pool = x.max(dim=1).values               # (B, lstm_hidden*2)
        x        = avg_pool + max_pool               # (B, lstm_hidden*2)

        # --- FC with skip connection ---
        skip = self.relu(self.fc_skip(x))            # (B, 128)

        x = self.relu(self.fc1_norm(self.fc1(x)))    # (B, 256)
        x = self.dropout(x)
        x = self.relu(self.fc2_norm(self.fc2(x)))    # (B, 128)
        x = x + skip                                  # skip connection
        x = self.dropout(x)
        x = self.fc3(x)                              # (B, 2)

        return x

# ============================================================
# STEP 7: Training with Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.best_state = None

    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter    = 0
            return False     # don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_state)
                return True  # stop
            return False

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for emb, lbl in loader:
        emb, lbl = emb.to(device), lbl.to(device)
        optimizer.zero_grad()
        out  = model(emb)
        loss = criterion(out, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == lbl).sum().item()
        total      += lbl.size(0)

    if scheduler:
        scheduler.step()
    return total_loss / len(loader), correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_preds, all_lbls = [], [], []

    for emb, lbl in loader:
        emb, lbl = emb.to(device), lbl.to(device)
        out  = model(emb)
        loss = criterion(out, lbl)

        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()

        total_loss += loss.item()
        correct    += (out.argmax(1) == lbl).sum().item()
        total      += lbl.size(0)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_lbls.extend(lbl.cpu().numpy())

    return (total_loss / len(loader), correct / total,
            np.array(all_preds), np.array(all_lbls), np.array(all_probs))

# ============================================================
# STEP 8: K-Fold Cross Validation Training
# ============================================================
def train_kfold(embeddings_np, labels_np, config):
    device  = config['device']
    skf     = StratifiedKFold(n_splits=config['k_folds'],
                               shuffle=True,
                               random_state=config['random_state'])

    fold_results = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings_np, labels_np)):
        print(f"\n{'='*55}")
        print(f"FOLD {fold+1}/{config['k_folds']}")
        print(f"{'='*55}")

        X_tr, X_val = embeddings_np[train_idx], embeddings_np[val_idx]
        y_tr, y_val = labels_np[train_idx],     labels_np[val_idx]

        # Weighted sampler for imbalance
        class_counts  = np.bincount(y_tr)
        sample_weights = 1.0 / class_counts[y_tr]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(y_tr), replacement=True
        )

        tr_ds  = PeptideDataset(torch.tensor(X_tr,  dtype=torch.float32),
                                 y_tr, augment=True)
        val_ds = PeptideDataset(torch.tensor(X_val, dtype=torch.float32),
                                 y_val, augment=False)

        tr_dl  = DataLoader(tr_ds,  batch_size=config['batch_size'],
                             sampler=sampler)
        val_dl = DataLoader(val_ds, batch_size=config['batch_size'],
                             shuffle=False)

        # Model
        model = ImprovedCNN_BiLSTM_ESM2(
            embed_dim      = config['embed_dim'],
            cnn_filters    = config['cnn_filters'],
            lstm_hidden    = config['lstm_hidden'],
            num_lstm_layers= config['num_lstm_layers'],
            dropout        = config['dropout'],
            num_classes    = 2
        ).to(device)

        # Class weights for focal loss
        cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        alpha    = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=alpha,
                               gamma=config['focal_gamma'],
                               label_smoothing=config['label_smoothing'])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            steps_per_epoch=len(tr_dl),
            epochs=config['epochs'],
            pct_start=0.2
        )
        early_stop = EarlyStopping(patience=15, min_delta=0.001)

        history = {'tr_loss':[], 'tr_acc':[], 'val_loss':[], 'val_acc':[]}

        for epoch in range(config['epochs']):
            tr_loss, tr_acc = train_epoch(
                model, tr_dl, optimizer, criterion, device, scheduler)
            val_loss, val_acc, preds, lbls, probs = eval_epoch(
                model, val_dl, criterion, device)

            history['tr_loss'].append(tr_loss)
            history['tr_acc'].append(tr_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                auc = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0
                print(f"  Epoch {epoch+1:3d} | "
                      f"TrAcc:{tr_acc:.4f} ValAcc:{val_acc:.4f} "
                      f"AUC:{auc:.4f}")

            if early_stop(val_acc, model):
                print(f"  Early stop at epoch {epoch+1}")
                break

        # Final fold evaluation
        _, val_acc, preds, lbls, probs = eval_epoch(
            model, val_dl, criterion, device)
        auc = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0

        print(f"\nFold {fold+1} → Val Acc: {val_acc:.4f} | AUC: {auc:.4f}")
        print(classification_report(lbls, preds,
              target_names=['WT', 'Mutant'], zero_division=0))

        fold_results.append({
            'fold': fold+1, 'acc': val_acc, 'auc': auc,
            'preds': preds, 'labels': lbls, 'probs': probs,
            'model_state': {k: v.clone() for k, v in model.state_dict().items()}
        })
        all_histories.append(history)

    return fold_results, all_histories

# ============================================================
# STEP 9: Plot All Results
# ============================================================
def plot_all_results(fold_results, all_histories, test_preds,
                     test_labels, test_probs):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # 1. K-Fold Accuracy Bar
    accs = [r['acc'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    axes[0,0].bar([f"Fold {r['fold']}" for r in fold_results],
                   accs, color='#1D9E75', alpha=0.8)
    axes[0,0].axhline(np.mean(accs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(accs):.4f}')
    axes[0,0].set_title('K-Fold Validation Accuracy')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].set_ylim(0, 1.1)
    axes[0,0].grid(alpha=0.3)

    # 2. K-Fold AUC Bar
    axes[0,1].bar([f"Fold {r['fold']}" for r in fold_results],
                   aucs, color='#534AB7', alpha=0.8)
    axes[0,1].axhline(np.mean(aucs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(aucs):.4f}')
    axes[0,1].set_title('K-Fold ROC-AUC')
    axes[0,1].set_ylabel('AUC')
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].grid(alpha=0.3)

    # 3. Training curves (best fold)
    best_fold_idx = np.argmax(accs)
    h = all_histories[best_fold_idx]
    axes[0,2].plot(h['tr_acc'],  label='Train Acc', color='#1D9E75')
    axes[0,2].plot(h['val_acc'], label='Val Acc',   color='#534AB7')
    axes[0,2].set_title(f'Training Curves (Best Fold {best_fold_idx+1})')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].legend()
    axes[0,2].grid(alpha=0.3)

    # 4. Loss curves
    axes[1,0].plot(h['tr_loss'],  label='Train Loss', color='#D85A30')
    axes[1,0].plot(h['val_loss'], label='Val Loss',   color='#BA7517')
    axes[1,0].set_title('Loss Curves (Best Fold)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # 5. Test Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['WT', 'Mutant'],
                yticklabels=['WT', 'Mutant'], ax=axes[1,1])
    axes[1,1].set_title('Test Confusion Matrix')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Actual')

    # 6. ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    auc = roc_auc_score(test_labels, test_probs)
    axes[1,2].plot(fpr, tpr, color='#1D9E75', lw=2,
                   label=f'AUC = {auc:.4f}')
    axes[1,2].plot([0,1],[0,1],'k--', lw=1)
    axes[1,2].set_title('ROC Curve (Test Set)')
    axes[1,2].set_xlabel('False Positive Rate')
    axes[1,2].set_ylabel('True Positive Rate')
    axes[1,2].legend()
    axes[1,2].grid(alpha=0.3)

    # 7. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(test_labels, test_probs)
    ap = average_precision_score(test_labels, test_probs)
    axes[2,0].plot(rec, prec, color='#534AB7', lw=2,
                   label=f'AP = {ap:.4f}')
    axes[2,0].set_title('Precision-Recall Curve')
    axes[2,0].set_xlabel('Recall')
    axes[2,0].set_ylabel('Precision')
    axes[2,0].legend()
    axes[2,0].grid(alpha=0.3)

    # 8. Probability Distribution
    wt_p  = test_probs[test_labels == 0]
    mut_p = test_probs[test_labels == 1]
    axes[2,1].hist(wt_p,  bins=30, alpha=0.6, color='#1D9E75', label='WT')
    axes[2,1].hist(mut_p, bins=30, alpha=0.6, color='#D85A30', label='Mutant')
    axes[2,1].axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
    axes[2,1].set_title('Prediction Probability Distribution')
    axes[2,1].set_xlabel('P(Mutant)')
    axes[2,1].legend()
    axes[2,1].grid(alpha=0.3)

    # 9. Per-class metrics summary
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        'Precision': [precision_score(test_labels, test_preds, pos_label=i)
                      for i in [0,1]],
        'Recall':    [recall_score(test_labels, test_preds, pos_label=i)
                      for i in [0,1]],
        'F1':        [f1_score(test_labels, test_preds, pos_label=i)
                      for i in [0,1]],
    }
    x = np.arange(2)
    w = 0.25
    for i, (m, v) in enumerate(metrics.items()):
        axes[2,2].bar(x + i*w, v, w, label=m, alpha=0.8)
    axes[2,2].set_xticks(x + w)
    axes[2,2].set_xticklabels(['WT', 'Mutant'])
    axes[2,2].set_title('Per-Class Metrics (Test Set)')
    axes[2,2].set_ylim(0, 1.2)
    axes[2,2].legend()
    axes[2,2].grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Improved CNN + Bi-LSTM + ESM-2 | p53 WT vs Mutant\n'
        f'Mean CV Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | '
        f'Mean AUC: {np.mean(aucs):.4f}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('improved_cnn_bilstm_esm2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → improved_cnn_bilstm_esm2.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = CONFIG['device']

    # 1. Load data
    print("="*55)
    print("STEP 1: Load Data")
    print("="*55)
    df = load_dataset(CONFIG['csv_path'])

    # 2. Augment minority class
    print("\n" + "="*55)
    print("STEP 2: Augmentation")
    print("="*55)
    df_aug = create_augmented_dataset(df, aug_factor=3, aug_prob=0.3)

    # Hold out test set BEFORE augmentation (no data leakage)
    df_orig = df.copy()
    X_orig  = df_orig['sequence'].tolist()
    y_orig  = df_orig['label'].tolist()

    _, X_test_seqs, _, y_test = train_test_split(
        X_orig, y_orig,
        test_size=CONFIG['test_size'],
        stratify=y_orig,
        random_state=CONFIG['random_state']
    )
    test_idx = set(zip(X_test_seqs, y_test))

    # Training pool = augmented data minus test sequences
    df_train = df_aug[~df_aug.apply(
        lambda r: (r['sequence'], r['label']) in test_idx, axis=1)]

    train_seqs   = df_train['sequence'].tolist()
    train_labels = df_train['label'].tolist()

    # 3. Load ESM-2
    print("\n" + "="*55)
    print("STEP 3: Load ESM-2")
    print("="*55)
    tokenizer = EsmTokenizer.from_pretrained(CONFIG['esm_model'])
    esm_model = EsmModel.from_pretrained(CONFIG['esm_model'])
    print(f"Loaded: {CONFIG['esm_model']}")

    # 4. Generate embeddings
    print("\n" + "="*55)
    print("STEP 4: ESM-2 Embeddings")
    print("="*55)
    all_seqs   = train_seqs + X_test_seqs
    all_embeds = get_esm2_embeddings(
        all_seqs, tokenizer, esm_model,
        device, batch_size=64, max_len=CONFIG['max_len']
    )
    n_train = len(train_seqs)
    train_embeds = all_embeds[:n_train].numpy()
    test_embeds  = all_embeds[n_train:].numpy()
    print(f"Train embeds: {train_embeds.shape}")
    print(f"Test embeds : {test_embeds.shape}")

    # 5. K-Fold Training
    print("\n" + "="*55)
    print("STEP 5: K-Fold Cross Validation Training")
    print("="*55)
    fold_results, all_histories = train_kfold(
        train_embeds, np.array(train_labels), CONFIG)

    # 6. Final test evaluation (best fold model)
    print("\n" + "="*55)
    print("STEP 6: Final Test Evaluation")
    print("="*55)
    best_fold = max(fold_results, key=lambda x: x['auc'])
    print(f"Using best fold: {best_fold['fold']} (AUC={best_fold['auc']:.4f})")

    final_model = ImprovedCNN_BiLSTM_ESM2(
        embed_dim      = CONFIG['embed_dim'],
        cnn_filters    = CONFIG['cnn_filters'],
        lstm_hidden    = CONFIG['lstm_hidden'],
        num_lstm_layers= CONFIG['num_lstm_layers'],
        dropout        = CONFIG['dropout']
    ).to(device)
    final_model.load_state_dict(best_fold['model_state'])

    test_ds = PeptideDataset(
        torch.tensor(test_embeds, dtype=torch.float32),
        np.array(y_test), augment=False
    )
    test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                          shuffle=False)

    cw = compute_class_weight('balanced',
         classes=np.unique(y_test), y=np.array(y_test))
    alpha     = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=CONFIG['focal_gamma'],
                           label_smoothing=0)

    _, test_acc, test_preds, test_labels, test_probs = eval_epoch(
        final_model, test_dl, criterion, device)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"ROC-AUC       : {roc_auc_score(test_labels, test_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
          target_names=['WT Peptide', 'Mutant Peptide']))

    torch.save(final_model.state_dict(), 'improved_cnn_bilstm_esm2_p53.pt')
    print("Model saved → improved_cnn_bilstm_esm2_p53.pt")

    # 7. Plot
    plot_all_results(fold_results, all_histories,
                      test_preds, test_labels, test_probs)

------------------------------------------------------------------------------------------------------------------------

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

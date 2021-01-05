import gzip
import Bio.SeqIO
import numpy as np
import torch

from torch.utils import data
from tokenizer import TAPETokenizer


class Dataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        # Return the size of the dataset
        return len(self.labels)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        x = self.inputs[index]
        y = self.labels[index]

        return x, y


def init_glyc(glyc_sites, length):
    """Encodes a sparse vector with protein length.
       glyc_sites set to None if protein is not a glycosylated protein
    """
    if glyc_sites == None:
        sparse_vector = np.zeros(length, dtype=int)
    else:
        sparse_vector = np.zeros(length, dtype=int)
        sparse_vector[glyc_sites] = 1
    return sparse_vector.astype(str)

def pad_sequences(sequences, max_length=2500, padding_value=0):
    shape = [len(sequences), max_length]
    dtype = sequences[0].dtype

    array = np.full(shape, padding_value, dtype=dtype)

    for i, sequence in enumerate(sequences):
        array[i][0:len(sequence)] = sequence

    return array


def tokenize_dataset(dataset, max_length=np.inf):
    """tokenizes dataset and splits according to glyc-type"""
    # encode data
    inputs_n = list()
    inputs_o = list()
    inputs_no_glyc_proteins = list()
    targets_n = list()
    targets_o = list()
    targets_no_glyc_proteins = list()
    inp_tokenizer = TAPETokenizer(vocab='iupac')
    tar_tokenizer = TAPETokenizer(vocab='glycolysation')

    for sequence_raw, features_raw in dataset:
        # tokenize protein sequence
        inp_tokenized = inp_tokenizer.encode(sequence_raw)
        if max_length > len(inp_tokenized):
            # find glycolysation positions
            glyc_sites_n = list()
            glyc_sites_o = list()
            for feature_raw in features_raw:
                if feature_raw.type == "glycosylation site":
                    if feature_raw.qualifiers["description"][0] == "N":
                        glyc_sites_n.append(feature_raw.location.end - 1)
                    elif feature_raw.qualifiers["description"][0] == "O":
                        glyc_sites_o.append(feature_raw.location.end - 1)           

            # make character vector and tokenizer
            if glyc_sites_n:
                glyc_state = init_glyc(glyc_sites_n, len(sequence_raw))
                targets_n.append(tar_tokenizer.encode(glyc_state))
                inputs_n.append(inp_tokenized)
            if glyc_sites_o:
                glyc_state = init_glyc(glyc_sites_o, len(sequence_raw))
                targets_o.append(tar_tokenizer.encode(glyc_state))
                inputs_o.append(inp_tokenized)
            
            #non glycosylated proteins
            if not glyc_sites_n and not glyc_sites_o:
                glyc_sites_none = None 
                glyc_state = init_glyc(glyc_sites_none, len(sequence_raw))
                targets_no_glyc_proteins.append(tar_tokenizer.encode(glyc_state))
                inputs_no_glyc_proteins.append(inp_tokenized)
                
            
    return inputs_n, inputs_o, inputs_no_glyc_proteins, targets_n, targets_o, targets_no_glyc_proteins


def construct_datasets(inputs, targets, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    """splits data into training, val and test"""
    # shuffle dataset TODO make shuffle code work
    rand_idx = torch.randperm(len(inputs))
    inputs = inputs[rand_idx]
    targets = targets[rand_idx]

    # Define partition sizes
    dataset_length = len(inputs)
    num_train = int(dataset_length * p_train)
    num_val = int(dataset_length * p_val)
    num_test = int(dataset_length * p_test)

    # Split sequences into partitions
    training_inp = inputs[: num_train]
    training_tar = targets[: num_train]
    validation_inp = inputs[num_train: num_train + num_val]
    validation_tar = targets[num_train: num_train + num_val]
    test_inp = inputs[-num_test:]
    test_tar = targets[-num_test:]

    training_set = dataset_class(training_inp, training_tar)
    validation_set = dataset_class(validation_inp, validation_tar)
    test_set = dataset_class(test_inp, test_tar)

    return training_set, validation_set, test_set


# read data from file
def load_unencoded_data(data_path):
    """file parser generator"""
    with gzip.open(data_path, 'rt') as f:
        for record in Bio.SeqIO.parse(f, format="uniprot-xml"): # uniprot-xml --> swiss
            yield record.seq, record.features


def onehot_convert_tensor(sequences, vocab_size=7):
    out = torch.zeros((sequences.numel(), vocab_size), dtype=torch.int64)
    out[torch.arange(sequences.numel()), sequences.view(-1)] = 1
    out = out.reshape(sequences.size() + (vocab_size,))
    return out


def predict_glycosylation(model, sequence, device):

    inp_tokenizer = TAPETokenizer(vocab='iupac')
    tar_tokenizer = TAPETokenizer(vocab='glycolysation')

    src = torch.tensor(inp_tokenizer.encode(sequence)).unsqueeze(0).to(device)

    tgt_list = tar_tokenizer.convert_tokens_to_ids(["<cls>", "<mask>"])

    for i in range(len(sequence)):

        tgt = torch.tensor(tgt_list).unsqueeze(0).to(device)

        tgt_key_padding_mask = torch.zeros(tgt.size(), dtype=torch.bool)
        tgt_key_padding_mask[0, -1] = True
        tgt_key_padding_mask.to(device)

        with torch.no_grad():
            output = model(src, tgt, tgt_key_padding_mask = tgt_key_padding_mask)

        best_guess = output[0,-1].argmax().item()
        tgt_list.append(best_guess)

    # remove start token
    return tgt_list[1:]


def load_checkpoint(model, optimizer, checkpoint_path, loss_data_path):
    """loads model state and returns training data"""
    train_loss, valid_loss = list(), list()
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if loss_data_path.is_file():
        data_load = np.load(loss_data_path, allow_pickle=True)
        train_loss = list(data_load["train_loss"])
        valid_loss = list(data_load["valid_loss"])
    return train_loss, valid_loss


def save_checkpoint(model, optimizer, train_loss, valid_loss, checkpoint_path, loss_data_path):
    """saves training data and model from a checkpoint file"""
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    np.savez_compressed(loss_data_path, train_loss=train_loss, valid_loss=valid_loss)


def write_test_set(dataset, data_path, glyc_type):
    """writes the test parition data to fasta and txt file to enable use in other models"""
    inp_tokenizer = TAPETokenizer(vocab='iupac')
    tar_tokenizer = TAPETokenizer(vocab='glycolysation')
    seqs = list()
    tars = list()
    special_tokens = ("<pad>", "<mask>", "<cls>", "<sep>", "<unk>")
    for i in range(len(dataset)):
        as_protein = inp_tokenizer.convert_ids_to_tokens(dataset.inputs[i].data.numpy())
        as_protein = "".join([x for x in as_protein if x not in special_tokens])
        header = ">test_seq_{}\n".format(i + 1)
        seq = header + as_protein
        seqs.append(seq)

        as_one_hot = tar_tokenizer.convert_ids_to_tokens(dataset.labels[i].data.numpy())
        as_one_hot = np.array([x for x in as_one_hot if x not in special_tokens], dtype=int)
        tars.append(as_one_hot)

    with open(data_path / "test_seqs_{}.fsa".format(glyc_type), "w") as seq_file:
        for seq, tar in zip(seqs, tars):
            print(seq, end="\n", file=seq_file)
    np.savez_compressed(data_path / "test_tars_{}.npz".format(glyc_type), tars)

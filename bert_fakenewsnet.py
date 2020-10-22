import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import os
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f'GPU count: {torch.cuda.device_count()}')

# reproducibility
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# training config
root = "/home/aibo/recent/FNID-dataset/FakeNewsNet"
max_seq_len = 512
learning_rate = 5e-5
batch_size = 32
epochs = 8
print(f"\nmax_seq_len = {max_seq_len}, learning_rate = {learning_rate}, batch_size = {batch_size}, epochs = {epochs}\n")

# # Load Dataset
file_names = ["fnn_train.csv", "fnn_dev.csv", "fnn_test.csv"]
train_path = os.path.join(root, file_names[0])
val_path = os.path.join(root, file_names[1])
test_path = os.path.join(root, file_names[2])

# In[ ]:
train_sample = pd.read_csv(train_path)
val_sample = pd.read_csv(val_path)
test_sample = pd.read_csv(test_path)

train_sample = train_sample.rename(columns={'label_fnn': 'label'})
val_sample = val_sample.rename(columns={'label_fnn': 'label'})
test_sample = test_sample.rename(columns={'label_fnn': 'label'})

train_sample['label'] = train_sample['label'].apply(lambda x: 1 if x == 'real' else 0)
val_sample['label'] = val_sample['label'].apply(lambda x: 1 if x == 'real' else 0)
test_sample['label'] = test_sample['label'].apply(lambda x: 1 if x == 'real' else 0)

train_sample = train_sample.drop(columns=['id', 'date', 'speaker', 'statement', 'sources', 'paragraph_based_content'])
val_sample = val_sample.drop(columns=['id', 'date', 'speaker', 'statement', 'sources', 'paragraph_based_content'])
test_sample = test_sample.drop(columns=['id', 'date', 'speaker', 'statement', 'sources', 'paragraph_based_content'])

train_text = train_sample.values.tolist()
val_text = val_sample.values.tolist()
test_text = test_sample.values.tolist()

train_labels = list(map(lambda x: x[1], train_text))
train_text = list(map(lambda x: x[0], train_text))
val_labels = list(map(lambda x: x[1], val_text))
val_text = list(map(lambda x: x[0], val_text))
test_labels = list(map(lambda x: x[1], test_text))
test_text = list(map(lambda x: x[0], test_text))

# # Import BERT Model and BERT Tokenizer
# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# sample data
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]
# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False)

# # Tokenization
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text,
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text,
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text,
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# # Convert Integer Sequences to Tensors
# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels)

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels)

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels)

print(f'shape of train val test set: {train_y.shape}, {val_y.shape}, {test_y.shape}')

# # Create DataLoaders
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

"""# # Freeze BERT Parameters
for param in bert.parameters():
    param.requires_grad = False"""

# # Define Model Architecture
class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc = nn.Sequential(
            nn.Linear(768, 2)
        )

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = cls_hs

        x = self.dropout(x)

        # output layer
        x = self.fc(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

# pass the pre-trained BERT to our define architecture and push the model to GPU
model = BERT_Arch(bert).cuda()
model = nn.DataParallel(model)

# optimizer from hugging face transformers
# define the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# define the scheduler
# scheduler = get_cosine_schedule_with_warmup(optimizer, )

# # Find Class Weights
# compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
print(f'class weights: {class_wts}')

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.cuda()

# loss function
cross_entropy = nn.NLLLoss(weight=weights)

# # Fine-Tune BERT
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def topk_accuracy(k, outputs, targets):
    """
    Compute top k accuracy
    """
    batch_size = targets.size(0)

    _, pred = outputs.topk(k, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.type(torch.FloatTensor).sum().item()

    return n_correct_elems / batch_size

def top1_accuracy(outputs, targets):
    return topk_accuracy(1, outputs, targets)

# function to train the model
def train():
    train_acc, val_acc, train_loss, val_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    print("\nTraining...")

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('Batch {} of {}: loss {:.5f}, acc {:.5f}'.format(step, len(train_dataloader), train_loss.avg,
                                                                   train_acc.avg))

        # push the batch to gpu
        batch = [r.cuda() for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        train_loss.update(loss.item())

        acc = top1_accuracy(preds, labels)
        train_acc.update(acc)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    print('Training finished: loss {:.5f}, acc {:.5f}'.format(train_loss.avg, train_acc.avg))

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate():
    train_acc, val_acc, train_loss, val_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 10 == 0 and not step == 0:
            # Report progress.
            print(
                'Batch {} of {}: loss {:.2f}, acc {:.5f}'.format(step, len(val_dataloader), val_loss.avg, val_acc.avg))

        # push the batch to gpu
        batch = [t.cuda() for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)
            val_loss.update(loss.item())

            acc = top1_accuracy(preds, labels)
            val_acc.update(acc)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    print('Evaluation finished: loss {:.2f}, acc {:.5f}'.format(val_loss.avg, val_acc.avg))

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# # Start Model Training
# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []

# for each epoch
best_model_name = 'saved_weights.pt'
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # train model
    t_loss, _ = train()

    # evaluate model
    v_loss, _ = evaluate()

    # save the best model
    if v_loss < best_valid_loss:
        best_valid_loss = v_loss
        torch.save(model.state_dict(), best_model_name)

    # append training and validation loss
    train_losses.append(t_loss)
    valid_losses.append(v_loss)

    print(f'\nTraining Loss: {t_loss:.6f}')
    print(f'Validation Loss: {v_loss:.6f}')

# torch.save(model.state_dict(), '/content/drive/My Drive/CS3244/models/BERT_L64_0.041.pt')

# In[ ]:
# load weights of best model
model.load_state_dict(torch.load(best_model_name))

# # Get Predictions for Test Data
# wrap tensors
total_preds, total_labels = [], []

print("\ntesting...")
for step, batch in enumerate(test_dataloader):
    if step % 50 == 0:
        print('Batch {} of {}'.format(step, len(test_dataloader)))

    # push the batch to gpu
    batch = [t.cuda() for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
        preds = model(sent_id, mask).detach().cpu().numpy()
        total_preds.append(preds)
        total_labels.append(labels.detach().cpu().numpy())

total_preds = np.concatenate(total_preds, axis=0)
total_labels = np.concatenate(total_labels, axis=0)

print(total_preds.shape, total_labels.shape)

preds = np.argmax(total_preds, axis=1)
print(classification_report(test_y, preds))

print(f'\nfinal test accuracy {np.sum(np.array(preds) == np.array(test_y)) / preds.shape[0]}\n')
print(f"\nmax_seq_len = {max_seq_len}, learning_rate = {learning_rate}, batch_size = {batch_size}, epochs = {epochs}\n")

# confusion matrix
pd.crosstab(test_y, preds)
print(pd)

# torch.save(model.state_dict(), '/content/drive/My Drive/CS3244/models/BERT_L64_best.pt')

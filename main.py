import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler


# Hyperparameters and constants
N_FEATURES = 2**12
N_EPOCHS = 2
CSV_SOURCE = 'News_dataset/'
MODEL_PATH = 'gru_model.pt'
USING_EXISTING_MODEL = True


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def train(train_loader, learn_rate, hidden_dim=256, epoch_count=N_EPOCHS):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2

    # Instantiating the models
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("\nStarting the GRU model training:\n")

    epoch_start_timestep = time.perf_counter()

    # Start training loop
    for epoch in range(1, epoch_count + 1):
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        epoch_end_timestep = time.perf_counter()
        print("Training epoch {}/{} completed. Total Loss: {:.8f}. Time Taken: {:.3f} seconds.".format(
            epoch, epoch_count, avg_loss / len(train_loader), epoch_end_timestep - epoch_start_timestep))
        epoch_start_timestep = epoch_end_timestep
    print("Training completed. Total time taken: {:.3f}".format(epoch_start_timestep - preprocessing_done))
    return model


def evaluate(model, test_x, test_y):
    number_of_test_cases = list(test_x.size())[0]
    print("\nTesting the model:\nNumber of test cases: {}\n".format(number_of_test_cases))
    evaluation_start = time.perf_counter()
    model.eval()
    outputs = []
    targets = []
    scaler = MinMaxScaler()
    scaler.fit(test_y)
    for i in range(number_of_test_cases):
        inp = torch.unsqueeze(torch.from_numpy(np.array(test_x[i])), dim=0)
        labs = torch.unsqueeze(torch.from_numpy(np.array(test_y[i])), dim=0)
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(round(out.item()))
        targets.append(round(labs.item()))
    accuracy = accuracy_score(targets, outputs)
    evaluation_end = time.perf_counter()
    print("Time taken: {:.3f} seconds".format(evaluation_end - evaluation_start))
    print("Accuracy: {}\n".format(accuracy))
    print("Sample outputs: {}".format(outputs[:10]))
    print("Respective targets: {}".format(targets[:10]))
    return outputs, targets, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting process...")
process_start = time.perf_counter()

print("Assigning truth targets to training data...")
true_data = pd.read_csv(CSV_SOURCE + 'True.csv')
true_data["is_fake"] = "0"
fake_data = pd.read_csv(CSV_SOURCE + 'Fake.csv')
fake_data["is_fake"] = "1"
dataset = pd.concat([true_data["text"], fake_data["text"]])
dataset.fillna(value="", inplace=True)
dataset_values = pd.concat([true_data["is_fake"], fake_data["is_fake"]])

print("Total entries found: {}\n".format(len(dataset)))

print("Splitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_values, train_size=0.8, random_state=0)
print("Number of training examples: {}".format(len(y_train)))
print("Number of testing examples: {}".format(len(y_test)))

print("Hashing input features...")
vectorizer = HashingVectorizer(n_features=N_FEATURES, ngram_range=(1, 3))
train_features = vectorizer.fit_transform(X_train).toarray()
test_features = vectorizer.fit_transform(X_test).toarray()

print("Loading data...")
X_train_tensor = torch.unsqueeze(torch.from_numpy(train_features), dim=1).float()
y_train_tensor = torch.unsqueeze(torch.from_numpy(y_train.to_numpy(dtype="float64")), dim=1).float()
X_test_tensor = torch.unsqueeze(torch.from_numpy(test_features), dim=1).float()
y_test_tensor = torch.unsqueeze(torch.from_numpy(y_test.to_numpy(dtype="float64")), dim=1).float()

batch_size = 1
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

preprocessing_done = time.perf_counter()
print("Time taken for pre-processing tasks: {:.3f} seconds".format(preprocessing_done - process_start))


if USING_EXISTING_MODEL:
    input_dim = next(iter(train_loader))[0].shape[2]
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    gru_model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    gru_model.load_state_dict(torch.load(MODEL_PATH))
    gru_outputs, targets, gru_sMAPE = evaluate(gru_model, X_test_tensor, y_test_tensor)
else:
    learning_rate = 0.001
    gru_model = train(train_loader, learning_rate)
    gru_outputs, targets, gru_sMAPE = evaluate(gru_model, X_test_tensor, y_test_tensor)
    torch.save(gru_model.state_dict(), MODEL_PATH)

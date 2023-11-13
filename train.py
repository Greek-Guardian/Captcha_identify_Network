import torch
import torch.nn
import time
from dataloader import get_dataloader
from model import captcha, init_weights, evaluate
from tabulate import tabulate

def train(opt):
    # device and opt.seed
    device = 'cuda' if torch.cuda.is_available() and opt.gpu else 'cpu'
    torch.manual_seed(opt.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(opt.seed)
        # torch.cuda.set_device('cuda')
        device = torch.device('cuda')
       
    # dataset loader
    print("Loading dataset to %s:" % device)
    dataloader = get_dataloader(opt.batch_size, shuffle=opt.shuffle,num_workers=opt.num_workers,\
                                device=device, dataset_dir_path=opt.dataset_dir_path, dataset_size=opt.dataset_size)
    print("Dataset loaded, num of batchs:", len(dataloader), ".")

    # init model
    if opt.load_or_not:
        try:
            model = torch.load(opt.model_load_path, map_location=device)
            print("Previous model read.")
        except:
            model = captcha().to(device)
            model.apply(init_weights)
            print("New model created.")
    else:
        model = captcha().to(device)
        model.apply(init_weights)
        print("New model created.")

    # define cost/loss & optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.BCELoss(reduce=False).to(device)

    # train my model
    total_batch = len(dataloader)
    model.train()    # set the model to train mode (dropout=True)
    print("{:-^71s}".format("Parameters:"))
    table_header = ['dataset_size', 'learning_rate', 'training_epochs', 'batch_size']
    a= [[opt.dataset_size, opt .learning_rate, opt.training_epochs, opt.batch_size]]
    print(tabulate(a, headers=table_header, tablefmt='grid'))
    print("{:-^71s}".format("Learning started. It takes sometime."))

    start_time = time.time()
    for epoch in range(opt.training_epochs):
        avg_cost = 0

        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            output= model(X)
            optimizer.zero_grad()
            cost = criterion(output.view(output.shape[0]*4, 62), Y.view(Y.shape[0]*4))
            cost.backward()
            optimizer.step()
            avg_cost += cost / total_batch

        torch.save(model, opt.model_save_path)
        print('[Epoch: {}/{}] Loss = {:>.9}'.format(epoch + 1, opt.training_epochs, avg_cost), ", Time consumption =", time.time()-start_time)
        start_time = time.time()
    # print(output.view(4, 62).shape, Y[0].shape)

    torch.save(model, opt.model_save_path)
    print("{:-^71s}".format("Learning Finished!"))

    # Test model and check accuracy
    with torch.no_grad():
        model.eval()    # set the model to evaluation mode (dropout=False)

        # get a batchsize num of samples
        for x, y in dataloader:
            x_data = x
            y_data = y
            break
        # X_test = x_data.view(len(x_data), 3, 200, 50).float().to(device)
        X_test = x_data.float().to(device)
        Y_test = y_data.to(device)

        prediction = model(X_test)
        correct_prediction = evaluate(prediction, Y_test)
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
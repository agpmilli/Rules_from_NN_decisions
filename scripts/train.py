import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter

def train(train_iter, dev_iter, model, writer, args):
    if args.cuda:
        model.cuda()

    if args.optim == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    steps = 0
    model.train()
    
    best_ever_test_accuracy, best_test_accuracy, best_test_loss = 0, 0, 0
    
    early_stop = args.early_stop
    current = 0
    
    avg_loss, avg_accuracy, loss_sum, corrects = 0, 0, 0, 0
    
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
    
        epoch_best_test_accuracy, epoch_best_test_loss = 0, 0
        
        print("EPOCH " + str(epoch))
        if current==early_stop:
            print("We didn't gain in test accuracy for " + str(early_stop) + " epochs -> we stop")
            break
            
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            
            loss = F.cross_entropy(logit, target)
            
            correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * correct/batch.batch_size
            
            loss_sum+=loss.data[0]
            corrects+=correct
            
            loss.backward()
            optimizer.step()           
            
            steps += 1
            
            if steps % args.log_interval == 0:
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             correct,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                test_accuracy, test_loss = eval(dev_iter, model, args)
                if test_accuracy > epoch_best_test_accuracy:
                    epoch_best_test_accuracy = test_accuracy
                    epoch_best_test_loss = test_loss
            
            if steps % (int)(len(dev_iter.dataset)/args.batch_size) == 0:
                break
        
        avg_loss = loss_sum/steps
        avg_accuracy = 100.0 * corrects/(steps*batch.batch_size)
        
        if epoch_best_test_accuracy > best_ever_test_accuracy:
            best_ever_test_accuracy = epoch_best_test_accuracy
            current=0
            torch.save(model.state_dict(), os.path.join(args.save_dir,"model"))
        else:
            current+=1
            
        print()
        print(avg_loss)
        print(avg_accuracy)
        print(epoch_best_test_loss)
        print(epoch_best_test_accuracy)   
        writer.add_scalar('train_loss', avg_loss, epoch) #data grouping by `slash`
        writer.add_scalar('train_accuracy', avg_accuracy, epoch)
        writer.add_scalar('eval_loss', epoch_best_test_loss, epoch)
        writer.add_scalar('eval_accuracy', epoch_best_test_accuracy, epoch) 
        
    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))                                                                                                                        
        
    return accuracy, avg_loss
                                                                       


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x =x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]
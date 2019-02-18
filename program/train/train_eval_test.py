

def test(test_loader, model,thresholder=0.5):
        csv_map = OrderedDict({'filename': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            # bs, ncrops, c, h, w = images.size()
            filepath = [os.path.basename(i) for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

            with torch.no_grad():
                y_pred = model(image_var)
                # sigmoid to prob
                sm = nn.Sigmoid()
                sm_out = sm(y_pred)       
            label_predicts = np.arange(28)[sm_out>=thresholder]
            
            for label_predict in label_predicts:
                str_predict_label = ''
                for i in range(28):
                    if label_predict[i] == 1:
                        str_predict_label = ' '.join(str(i))
                    
                csv_map['probability'].append(str_predict_label)
            csv_map['filename'].extend(filepath)
            
            
        result = pd.DataFrame(csv_map)
    

        
        # 生成结果文件，保存在result文件夹中，可用于直接提交
        submission = pd.DataFrame(csv_map)
        submission.to_csv('./result/%s/submission.csv' % file_name, header=['Id','Predicted'], index=False)

#         prob_map = pd.DataFrame(prob_list)
#         prob_map.to_csv('./result/%s/prob_map.csv' % file_name, header=False, index=False)
        return



def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        
        # switch to train mode
        model.train()
        
        end = time.time()
        # 从训练集迭代器中获取训练数据
        for i, (images, target) in enumerate(train_loader):
            # 评估图片读取耗时
            data_time.update(time.time() - end)
            # 将图片和标签转化为tensor
            image_var = torch.tensor(images).cuda(async=True)
            label = torch.tensor(target).cuda(async=True)
            
            # 将图片输入网络，前传，生成预测值
            y_pred = model(image_var)
            # 计算loss
            loss = criterion(y_pred, label)
            losses.update(loss.item(), images.size(0))

            # 计算正确率
            prec, PRED_COUNT = accuracy(y_pred.data, target)
            acc.update(prec, PRED_COUNT)

            # 对梯度进行反向传播，使用随机梯度下降更新网络权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 评估训练耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 打印耗时与结果
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))


def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            image_var = torch.tensor(images).cuda(async=True)
            target = torch.tensor(labels).cuda(async=True)

            # 图片前传。验证和测试时不需要更新网络权重，所以使用torch.no_grad()，表示不计算梯度
            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)
            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, labels)
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i == 0:
                total_y_pred = y_pred
                total_labels = labels
            else:
                total_y_pred = torch.cat((total_y_pred,y_pred),0)
                total_labels = torch.cat((total_labels,labels),0)
            if i % print_freq == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
        print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        return acc.avg, losses.avg,total_y_pred,total_labels
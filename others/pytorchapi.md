- ### 分布式训练

  之前有model  = DataParallel(model.cuda(),,device),现在基本上不用这个，原因是其只支持单进程，效率太慢，同时也不支持多机情况，也不支持模型并行，模型的保存和加载

​		torch.save 注意需要调用model.module.state_dict(), torch.load 需要注意map_location的使用

​	
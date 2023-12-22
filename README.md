
**First**, run the command below to get the splited dataset MNIST:
'''sh
# generate the splited dataset
python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 10
'''

**Second**, run the command below to qucikly get a result of the basic algorithm FedAVE on MNIST with a simple MLP:
'''sh
python main.py --task mnist_cnum10_dist18_skew0.0_seed0 --model mlp --algorithm FedAVE --num_rounds 20
--num_epochs 3 --learning_rate 0.15 --batch_size 32 --eval_interval 1 --Beta 5 --gpu 0
'''
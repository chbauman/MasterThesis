# (Train and) Evaluate trained agents

# Trained on Euler
python .\BatchRL.py -v -r --train_data train_val --fl 50.0 21.0 25.0 -in 2000000 -bo f t --room_nr 43

# Trained on Remote
python .\BatchRL.py -v -r --train_data all --fl 50.0 --rl_sampling all --hop_eval_data test -in 500000 -bo f t --room_nr 43
python .\BatchRL.py -v -r --train_data all --fl 50.0 --rl_sampling all --hop_eval_data test -in 500000 -bo f t --room_nr 41

# Cleanup
python .\BatchRL.py -v -c

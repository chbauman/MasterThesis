# (Train and) Evaluate trained agents
python .\BatchRL.py -v -r --train_data all --fl 50.0 --rl_sampling all --hop_eval_data test -in 500000 -bo f t --room_nr 43
python .\BatchRL.py -v -r --train_data all --fl 50.0 --rl_sampling all --hop_eval_data test -in 500000 -bo f t --room_nr 41

# Cleanup
python .\BatchRL.py -v -c

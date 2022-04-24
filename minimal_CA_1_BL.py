#!/usr/bin/env python

import sys
import argparse
import json
import time

import numpy as np
import gym

import brica1
import brica1.brica_gym

import brical
import minimal_CA_1 as mca1

import torch

from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder


def main():
    parser = argparse.ArgumentParser(description='BriCA Minimal Cognitive Architecture with Gym')
    parser.add_argument('mode', help='1:random act, 2: reinforcement learning', choices=['1', '2'])
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                        help='Max steps in an episode (default: 20)')
    parser.add_argument('--config', type=str, default='minimal_CA.json', metavar='N',
                        help='Model configuration (default: minimal_CA.json')
    parser.add_argument('--model', type=str, metavar='N',
                        help='Saved model for visual path')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--brical',  type=str, default='minimalCA.brical.json', metavar='N',
                        help='a BriCAL json file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    nb = brical.NetworkBuilder()
    f = open(args.brical)
    nb.load_file(f)
    if not nb.check_consistency():
        sys.stderr.write("ERROR: " + args.brical + " is not consistent!\n")
        exit(-1)

    if not nb.check_grounding():
        sys.stderr.write("ERROR: " + args.brical + " is not grounded!\n")
        exit(-1)

    observation_dim = config['env']['observation_dim']
    input_shape = [-1, observation_dim]
    motor_obs_dim = config["motor_obs_dim"]
    
    env = gym.make(config['env']['name'])
    train = {"episode_count": args.episode_count, "max_steps": args.max_steps, 'rl_agent': config['rl_agent']}

    if args.dump is not None:
        try:
            observation_dump = open(args.dump, mode='w')
        except OSError as e:
            sys.stderr.write(str(e) + '\n')
            sys.exit(1)
    else:
        observation_dump = None

    nb.unit_dic['minimalCA.VisualComponent'].__init__(observation_dim, motor_obs_dim)
    nb.unit_dic['minimalCA.MotorComponent'].__init__(observation_dim, config["motor_n_action"], True, train)

    nb.make_ports()

    if args.model is not None:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if config['visual']['model'] == 'SimpleAE':
            visual_model = SimpleAutoencoder(input_shape, config['visual']['model_config']).to(device)
        else:
            raise NotImplementedError('Model not supported: ' + str(config['model']))
        visual_model.load_state_dict(torch.load(args.model))
        visual_model.eval()
        nb.unit_dic['minimalCA.VisualComponent'].device = device
    else:
        visual_model = None
    nb.unit_dic['minimalCA.VisualComponent'].model = visual_model

    agent_builder = brical.AgentBuilder()
    model = nb.unit_dic['minimalCA.CognitiveArchitecture']
    agent = agent_builder.create_gym_agent(nb, model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    for i in range(train["episode_count"]):
        reward_sum = 0.
        last_token = 0
        for j in range(train["max_steps"]):
            scheduler.step()
            nb.unit_dic['minimalCA.VisualComponent'].inputs['token_in'] = model.get_out_port('token_out').buffer
            time.sleep(config["sleep"])
            current_token = agent.get_out_port('token_out').buffer[0]
            if last_token + 1 == current_token:
                reward_sum += agent.get_in_port("reward").buffer[0]
                last_token = current_token
                env.render()
                if observation_dump is not None:
                    observation_dump.write(str(agent.get_in_port("observation").buffer.tolist()) + '\n')
            if agent.env.done:
                agent.env.flush = True
                scheduler.step()
                while agent.get_in_port('token_in').buffer[0] != agent.get_out_port('token_out').buffer[0]:
                    scheduler.step()
                agent.env.reset()
                nb.unit_dic['minimalCA.MotorComponent'].reset()
                nb.unit_dic['minimalCA.VisualComponent'].results['token_out'] = np.array([0])
                nb.unit_dic['minimalCA.VisualComponent'].out_ports['token_out'].buffer = np.array([0])
                break
        print(i, "Avr. reward: ", reward_sum/env.spec.max_episode_steps)

    print("Close")
    if observation_dump is not None:
        observation_dump.close()
    env.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import sys
import argparse
import json
import time

import gym

import logging
import numpy as np

import brica1
import brica1.brica_gym

import torch

from tensorforce.environments import Environment
from tensorforce.agents import Agent

from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder


class PipeVisualComponent(brica1.PipeComponent):
    def __init__(self):
        super(PipeVisualComponent, self).__init__()
        self.make_in_port('in', 1)
        self.make_out_port('out', 1)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.set_map("in", "out")
        self.set_map("token_in", "token_out")


class ModeledVisualComponent(brica1.brica_gym.Component):
    def __init__(self, in_dim, out_dim):
        super(ModeledVisualComponent, self).__init__()
        self.make_in_port('in', in_dim)
        self.make_out_port('out', out_dim)
        self.model = None
        self.device = None
        self.in_dim = in_dim
        self.out_dim = out_dim

    def fire(self):
        in_data = self.get_in_port('in').buffer
        x = in_data.reshape(1, self.in_dim)
        data = torch.from_numpy(x.astype(np.float64)).float().to(self.device)
        encoding, _ = self.model(data)
        self.results['out'] = encoding.to(self.device).detach().numpy().reshape(self.in_dim,)


class MotorComponent(brica1.brica_gym.Component):
    class MotorEnv(Environment):
        def __init__(self, n_action, obs_dim, parent):
            super(Environment, self).__init__()
            self.state_space = dict(type='float', shape=(obs_dim,))
            self.action_space = dict(type='int', num_values=n_action)
            self.state = np.random.random(size=(obs_dim,))
            self.reward = 0.0
            self.done = False
            self.info = {}
            self.parent = parent

        def states(self):
            return self.state_space

        def actions(self):
            return self.action_space

        def reset(self):
            self.state = np.random.random(size=self.state_space['shape'])
            return self.state

        def execute(self, actions):
            if not isinstance(self.parent.get_in_port('observation').buffer[0], np.int16):
                self.state = self.parent.get_in_port('observation').buffer
            reward = self.parent.get_in_port('reward').buffer[0]
            done = self.parent.get_in_port('done').buffer[0]
            if done == 1:
                done = True
            else:
                done = False
            return self.state, done, reward

    def __init__(self, in_dim, n_action, rl, train):
        super().__init__()
        self.make_in_port('observation', in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', 1)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.n_action = n_action    # number of action choices
        self.results['action'] = np.array([np.random.randint(n_action)])
        self.model = None
        self.env_type = "MotorEnv"
        self.token = 0
        self.prev_actions = 0
        self.init = True
        self.in_dim = in_dim
        self.rl = rl
        if rl:
            self.env = Environment.create(environment=MotorComponent.MotorEnv,
                                          max_episode_timesteps=train["episode_count"]*train["max_steps"],
                                          n_action=n_action, obs_dim=in_dim, parent=self)
            self.env.reset()
            self.agent = Agent.create(agent=train['rl_agent'], environment=self.env)

    def fire(self):
        if self.rl:
            if self.init:
                state = self.get_in_port('observation').buffer
                self.act(state)
                self.init = False
            state, terminal, reward = self.env.execute(actions=self.prev_actions)
            # print("TF:", self.token, state, reward, terminal, self.prev_actions)
            self.agent.timestep_completed[:] = False
            self.agent.observe(terminal=terminal, reward=reward)
            if not terminal:
                self.act(state)
        else:
            self.results['action'] = np.array([np.random.randint(self.n_action)])

    def act(self, state):
        actions = self.agent.act(states=state)
        self.prev_actions = actions
        if np.isscalar(actions):
            self.results['action'] = np.array([actions])
        else:
            self.results['action'] = np.array([actions[0]])
    
    def reset(self):
        self.token = 0
        self.init = True
        if self.rl:
            self.env.reset()
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.out_ports['token_out'].buffer = np.array([0])


class CognitiveArchitecture(brica1.Module):
    def __init__(self, observation_dim, motor_obs_dim, n_action, rl, train, modelp):
        super(CognitiveArchitecture, self).__init__()
        self.make_in_port('observation', observation_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', 1)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        if modelp:
            self.visual = ModeledVisualComponent(observation_dim, motor_obs_dim)
        else:
            self.visual = PipeVisualComponent()
        self.motor = MotorComponent(observation_dim, n_action, rl, train)
        self.add_component('visual', self.visual)
        self.add_component('motor', self.motor)
        self.visual.alias_in_port(self, 'observation', 'in')
        self.visual.alias_in_port(self, 'token_in', 'token_in')
        self.motor.alias_in_port(self, 'reward', 'reward')
        self.motor.alias_in_port(self, 'done', 'done')
        self.motor.alias_out_port(self, 'action', 'action')
        self.motor.alias_out_port(self, 'token_out', 'token_out')
        brica1.connect((self.visual, 'out'), (self.motor, 'observation'))
        brica1.connect((self.visual, 'token_out'), (self.motor, 'token_in'))


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
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    observation_dim = config['env']['observation_dim']
    input_shape = [-1, observation_dim]
    motor_obs_dim = config["motor_obs_dim"]
    
    env = gym.make(config['env']['name'])
    train = {"episode_count": args.episode_count, "max_steps": args.max_steps}

    if args.dump is not None:
        try:
            observation_dump = open(args.dump, mode='w')
        except OSError as e:
            sys.stderr.write(str(e) + '\n')
            sys.exit(1)
    else:
        observation_dump = None

    md = args.mode
    model = None
    if md == "1":    # random act
        model = CognitiveArchitecture(observation_dim, motor_obs_dim, config["motor_n_action"], False, train, False)
    elif md == "2":  # act by reinforcement learning
        train['rl_agent'] = config['rl_agent']
        modelp = args.model is not None
        model = CognitiveArchitecture(observation_dim, motor_obs_dim, config["motor_n_action"], True, train, modelp)
        if modelp:
            use_cuda = not args.no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            if config['visual']['model'] == 'SimpleAE':
                visual_model = SimpleAutoencoder(input_shape, config['visual']['model_config']).to(device)
            else:
                raise NotImplementedError('Model not supported: ' + str(config['model']))
            visual_model.load_state_dict(torch.load(args.model))
            visual_model.eval()
            model.visual.device = device
        else:
            visual_model = None
        model.visual.model = visual_model

    agent = brica1.brica_gym.GymAgent(model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    for i in range(train["episode_count"]):
        reward_sum = 0.
        last_token = 0
        for j in range(train["max_steps"]):
            scheduler.step()
            model.visual.inputs['token_in'] = model.get_out_port('token_out').buffer
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
                model.motor.reset()
                model.visual.results['token_out'] = np.array([0])
                model.visual.out_ports['token_out'].buffer = np.array([0])
                break
        print(i, "Avr. reward: ", reward_sum/env.spec.max_episode_steps)

    print("Close")
    if observation_dump is not None:
        observation_dump.close()
    env.close()


if __name__ == '__main__':
    main()

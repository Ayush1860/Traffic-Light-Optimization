import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs, FuzzyEvaluator):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._FuzzyEvaluator = FuzzyEvaluator
        
        from simulation.strategy import MaxPressureStrategy
        self._Strategy = MaxPressureStrategy()


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # ---- REWARD CALCULATION ----
            # Primary reward: change in total waiting time (positive = improvement)
            current_total_wait = self._collect_waiting_times()
            wait_reward = old_total_wait - current_total_wait

            # Secondary: Fuzzy congestion modifier (small penalty for high congestion)
            density = self._get_avg_density()
            queue_lengths = self._get_queue_lengths_dict()
            total_queue = sum(queue_lengths.values())
            congestion_score = self._FuzzyEvaluator.get_congestion_score(density, total_queue)
            
            # Combined reward: wait-time improvement minus gentle congestion nudge
            # congestion_score is 0-1, kept small so wait_reward drives learning
            reward = wait_reward - (congestion_score * 0.5)

            # Determine if this is the last step (terminal state)
            is_done = (self._step + self._green_duration + self._yellow_duration) >= self._max_steps

            # saving the data into the memory (with done flag for proper Q-learning)
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state, is_done))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # execute the phase selected before
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action, action)
                self._simulate(self._yellow_duration)
            
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # track total reward (both positive and negative)
            self._sum_reward += reward

        self._save_episode_stats()
        print("Total reward:", round(self._sum_reward, 2), "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["W2TL1", "N12TL1", "S12TL1", "TL22TL1", 
                          "TL12TL2", "N22TL2", "S22TL2", "TL32TL2", 
                          "TL22TL3", "N32TL3", "S32TL3", "E2TL3"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _decode_action(self, action):
        return action // 4, (action // 2) % 2, action % 2

    def _set_yellow_phase(self, old_action, new_action):
        """
        Activate the correct yellow light combination in sumo
        """
        old_a1, old_a2, old_a3 = self._decode_action(old_action)
        new_a1, new_a2, new_a3 = self._decode_action(new_action)
        
        phase1 = old_a1 * 2 + 1 if old_a1 != new_a1 else old_a1 * 2
        phase2 = old_a2 * 2 + 1 if old_a2 != new_a2 else old_a2 * 2
        phase3 = old_a3 * 2 + 1 if old_a3 != new_a3 else old_a3 * 2

        traci.trafficlight.setPhase("TL1", phase1)
        traci.trafficlight.setPhase("TL2", phase2)
        traci.trafficlight.setPhase("TL3", phase3)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        a1, a2, a3 = self._decode_action(action_number)
        traci.trafficlight.setPhase("TL1", a1 * 2)
        traci.trafficlight.setPhase("TL2", a2 * 2)
        traci.trafficlight.setPhase("TL3", a3 * 2)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        q_dict = self._get_queue_lengths_dict()
        return sum(q_dict.values())

    def _get_queue_lengths_dict(self):
        """
        Retrieve queue length per incoming lane for Max Pressure
        """
        incoming_roads = ["W2TL1", "N12TL1", "S12TL1", "TL22TL1", 
                          "TL12TL2", "N22TL2", "S22TL2", "TL32TL2", 
                          "TL22TL3", "N32TL3", "S32TL3", "E2TL3"]
        q_dict = {}
        for road in incoming_roads:
            q_dict[road] = traci.edge.getLastStepHaltingNumber(road)
        return q_dict

    def _get_avg_density(self):
        """
        Retrieve average density of incoming lanes (0-1) for Fuzzy Logic
        """
        incoming_roads = ["W2TL1", "N12TL1", "S12TL1", "TL22TL1", 
                          "TL12TL2", "N22TL2", "S22TL2", "TL32TL2", 
                          "TL22TL3", "N32TL3", "S32TL3", "E2TL3"]
        total_density = 0
        MAX_CAPACITY = 100 # Approx vehicle capacity per lane (750m / 7.5m)
        
        for road in incoming_roads:
            count = traci.edge.getLastStepVehicleNumber(road)
            density = min(count / MAX_CAPACITY, 1.0)
            total_density += density
            
        return total_density / len(incoming_roads)


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            if lane_id.startswith("W2TL1_"):
                lane_group = 0 if lane_id in ["W2TL1_0", "W2TL1_1", "W2TL1_2"] else 1
            elif lane_id.startswith("N12TL1_"):
                lane_group = 2 if lane_id in ["N12TL1_0", "N12TL1_1", "N12TL1_2"] else 3
            elif lane_id.startswith("TL22TL1_"):
                lane_group = 4 if lane_id in ["TL22TL1_0", "TL22TL1_1", "TL22TL1_2"] else 5
            elif lane_id.startswith("S12TL1_"):
                lane_group = 6 if lane_id in ["S12TL1_0", "S12TL1_1", "S12TL1_2"] else 7
            elif lane_id.startswith("TL12TL2_"):
                lane_group = 8 if lane_id in ["TL12TL2_0", "TL12TL2_1", "TL12TL2_2"] else 9
            elif lane_id.startswith("N22TL2_"):
                lane_group = 10 if lane_id in ["N22TL2_0", "N22TL2_1", "N22TL2_2"] else 11
            elif lane_id.startswith("TL32TL2_"):
                lane_group = 12 if lane_id in ["TL32TL2_0", "TL32TL2_1", "TL32TL2_2"] else 13
            elif lane_id.startswith("S22TL2_"):
                lane_group = 14 if lane_id in ["S22TL2_0", "S22TL2_1", "S22TL2_2"] else 15
            elif lane_id.startswith("TL22TL3_"):
                lane_group = 16 if lane_id in ["TL22TL3_0", "TL22TL3_1", "TL22TL3_2"] else 17
            elif lane_id.startswith("N32TL3_"):
                lane_group = 18 if lane_id in ["N32TL3_0", "N32TL3_1", "N32TL3_2"] else 19
            elif lane_id.startswith("E2TL3_"):
                lane_group = 20 if lane_id in ["E2TL3_0", "E2TL3_1", "E2TL3_2"] else 21
            elif lane_id.startswith("S32TL3_"):
                lane_group = 22 if lane_id in ["S32TL3_0", "S32TL3_1", "S32TL3_2"] else 23
            else:
                lane_group = -1

            if lane_group >= 0:
                car_position = lane_group * 10 + lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _, done = b[0], b[1], b[2], b[3], b[4]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                if done:
                    current_q[action] = reward  # terminal state: no future reward
                else:
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_reward)  # total reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

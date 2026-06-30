import gymnasium as gym 
import numpy as np

class HP:
  def __init__( self, nsteps=1000, eplength=2000, lr=0.02, deltas=16,bdeltas=16, noise=0.03, seed=1, envn="BippedalWalker-v3",):
    self.nsteps=nsteps
    self.eplegth=eplength
    self.lr=lr
    self.deltas=deltas
    sel.bdeltas=bdeltas
    self.noise=noise
    self.seed=seed
    self.envn=envn

class Normalizer:
  def __init__(self,ninptus):
    self.n=np.zeros(ninputs)
    self.mean=np.zeros(ninputs)
    self.meandif=np.zeros(ninputs)
    self.var=np.zeros(ninputs)
  def observe(self, x):
      self.n += 1.0
      lastMean = self.mean.copy()
      self.mean += (x - self.mean) / self.n
      self.meanDiff += (x - lastMean) * (x - self.mean)
      self.var = (self.meanDiff / self.n).clip(min=1e-2)
  def normalize(self, inputs):
      return (inputs - self.mean) / np.sqrt(self.var)

class Policy:
    def __init__(self, inputSize, outputSize, hp):
        self.theta = np.zeros((outputSize, inputSize))
        self.hp = hp
    def evaluate(self, inputs, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(inputs)
        if direction == "+":
            return (self.theta + self.hp.noise * delta).dot(inputs)
        return (self.theta - self.hp.noise * delta).dot(inputs)
    def sampleDeltas(self):
        return [ np.random.randn(*self.theta.shape) for p in range(self.hp.numDeltas)]
    def update(self, rollouts, sigmaRewards):
        step = np.zeros_like(self.theta)
        for positiveReward, negativeReward, delta in rollouts:
            step += (positiveReward - negativeReward) * delta
        self.theta += (self.hp.learningRate / (self.hp.numBestDeltas * sigmaRewards)* step )

class ARSTrainer:
    def __init__(self,hp=None,inputSize=None,outputSize=None,normalizer=None,policy=None,):
        self.hp = hp or HP()
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.envn, render_mode="human")
        self.hp.episodeLength = (self.env.spec.max_episode_stepsor self.hp.episodeLength)
        self.inputSize = (inputSizeor self.env.observation_space.shape[0])
        self.outputSize = (outputSizeor self.env.action_space.shape[0])
        self.normalizer = (normalizeror Normalizer(self.inputSize))
        self.policy = (policyor Policy( self.inputSize, self.outputSize,self.hp,))
    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        done = False
        totalReward = 0.0
        numPlays = 0
        while not done and numPlays < self.hp.episodeLength:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state,delta,direction,  )
            result = self.env.step(action)
            if len(result) == 5:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                state, reward, done, _ = result
            reward = np.clip(reward, -1, 1)
            totalReward += reward
            numPlays += 1
        return totalReward
    def train(self):
        for step in range(self.hp.nbSteps):
            deltas = self.policy.sampleDeltas()
            positiveRewards = [0] * self.hp.numDeltas
            negativeRewards = [0] * self.hp.numDeltas
            for i in range(self.hp.numDeltas):
                positiveRewards[i] = self.explore("+", deltas[i])
                negativeRewards[i] = self.explore("-", deltas[i])
            sigmaRewards = np.std(positiveRewards + negativeRewards)
            scores = { i: max(positiveRewards[i],negativeRewards[i],) for i in range(self.hp.numDeltas)}
            order = sorted(scores,key=scores.get,reverse=True,)[: self.hp.numBestDeltas]
            rollouts = [(positiveRewards[i],negativeRewards[i],deltas[i],for i in order]
            self.policy.update(rollouts,sigmaRewards,)
            reward = self.explore()
            print( f"Step {step} | Reward {reward:.2f}")
        self.env.close()
hp = HP(envName="BipedalWalker-v3")
trainer = ARSTrainer(hp=hp)
trainer.train()



  
 
    

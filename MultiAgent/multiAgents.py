# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"


        food2 = currentGameState.getFood()
        currPos = list(successorGameState.getPacmanPosition())
        maxDistance = -999999
        distance = 0
        foodsList = food2.asList()

        if action == 'Stop':
            return -999999

        for state in newGhostStates:
            if state.getPosition() == tuple(currPos) and (state.scaredTimer == 0):
                return -999999

        for food in foodsList:
            distance = -1 * (manhattanDistance(food, currPos))

            if distance > maxDistance:
                maxDistance = distance

        return maxDistance


        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minMax(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness = deepness + 1
            if deepness == self.depth or gameState.isWin() or gameState.isLose():
                "only one depth or we're close to winning/losing"
                return self.evaluationFunction(gameState)
            elif agent == 0:
                return max(gameState, deepness, agent)
            else:
                return min(gameState, deepness, agent)

        def max(gameState, deepness, agent):
            output = ["", -float("inf")]
            "output has one placeholder for action and one for value which we currently set as minus infinity"
            pacmanActions = gameState.getLegalActions(agent)

            if not pacmanActions:
                return self.evaluationFunction(gameState)

            for action in pacmanActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMax(currState, deepness, agent + 1)
                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue
                if val > output[1]:
                    output = [action, val]
            return output

        def min(gameState, deepness, agent):
            output = ["", float("inf")]
            "output has one placeholder for action and one for value which we currently set as minus infinity"
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMax(currState, deepness, agent + 1)
                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue
                if val < output[1]:
                    output = [action, val]
            return output

        outputList = minMax(gameState, 0, 0)
        return outputList[0]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minMax(gameState, deepness, agent, alpha, beta):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness = deepness + 1
            if deepness == self.depth or gameState.isWin() or gameState.isLose():
                "only one depth or we're close to winning/losing"
                return self.evaluationFunction(gameState)
            elif agent == 0:
                return maxFun(gameState, deepness, agent, alpha, beta)
            else:
                return minFun(gameState, deepness, agent, alpha, beta)

        def maxFun(gameState, deepness, agent, alpha, beta):
            output = ["", -float("inf")]
            "output has one placeholder for action and one for value which we currently set as minus infinity"
            pacmanActions = gameState.getLegalActions(agent)

            if not pacmanActions:
                return self.evaluationFunction(gameState)

            for action in pacmanActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMax(currState, deepness, agent + 1, alpha, beta)

                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue

                if val > output[1]:
                    output = [action, val]
                if val > beta:
                    return [action, val]
                alpha = max(alpha, val)
            return output

        def minFun(gameState, deepness, agent, alpha, beta):
            output = ["", float("inf")]
            "output has one placeholder for action and one for value which we currently set as minus infinity"
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMax(currState, deepness, agent + 1, alpha, beta)

                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue

                if val < output[1]:
                    output = [action, val]
                if val < alpha:
                    return [action, val]
                beta = min(beta, val)
            return output

        outputList = minMax(gameState, 0, 0, -float("inf"), float("inf"))
        return outputList[0]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness = deepness + 1
            if deepness == self.depth or gameState.isWin() or gameState.isLose():
                "only one depth or we're close to winning/losing"
                return self.evaluationFunction(gameState)
            elif agent == 0:
                return max(gameState, deepness, agent)
            else:
                return expectimax2(gameState, deepness, agent)

        def max(gameState, deepness, agent):
            output = ["", -float("inf")]
            "output has one placeholder for action and one for value which we currently set as minus infinity"
            pacmanActions = gameState.getLegalActions(agent)

            if not pacmanActions:
                return self.evaluationFunction(gameState)

            for action in pacmanActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = expectimax(currState, deepness, agent + 1)
                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue
                if val > output[1]:
                    output = [action, val]
            return output

        def expectimax2(gameState, deepness, agent):
            output = ["", 0]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            probability = 1.0 / len(ghostActions)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = expectimax(currState, deepness, agent + 1)
                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue
                output[0] = action
                output[1] += val * probability
            return output

        outputList = expectimax(gameState, 0, 0)
        return outputList[0]

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <We get the locations of Pacman, the food, capsules and ghosts. Then, using ManhattanDistance, we find what's the closest food(max of a small distance, which we turned to negative), avoiding ghosts ht high priority!>
    """
    "*** YOUR CODE HERE ***"
    "Get position of the food, capsules, pacman and the ghosts"
    foodPos = currentGameState.getFood().asList()
    foodDistance = []
    capPos = currentGameState.getCapsules()
    currPos = list(currentGameState.getPacmanPosition())
    ghostStates = currentGameState.getGhostStates()
    "Check distance to food from Pacman's current position"
    for food in foodPos:
        food2pacmanDist = manhattanDistance(food, currPos)
        foodDistance.append(-1 * food2pacmanDist)
    "Pacman reached his meal(same position)"
    if not foodDistance:
        foodDistance.append(0)
    "return closest food and which will give the highest score"
    return max(foodDistance) + currentGameState.getScore()

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction



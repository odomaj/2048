from agents.learning_agent import LearningAgent
import tensorflow as tf


def main() -> None:
    # Initialize the learning agent
    agent = LearningAgent()

    # Set up the training parameters
    num_epochs = 1000  # Set the number of epochs as needed for training
    print("Starting training...")

    # Train the agent
    agent.train(num_epochs)

    print("Training completed. Evaluating the agent...")

    """
    # Test the agent's performance
    total_rewards = 0
    num_episodes = 10
    for episode in range(num_episodes):
        time_step = agent.eval_env.reset()
        episode_reward = 0

        while not time_step.is_last():
            # Select action based on the policy for evaluation (greedy)
            action_step = agent.agent.policy.action(time_step)
            time_step = agent.eval_env.step(action_step.action)
            episode_reward += time_step.reward

        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    average_reward = total_rewards / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")
    """


if __name__ == "__main__":
    main()

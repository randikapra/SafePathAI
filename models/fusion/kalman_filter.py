import numpy as np

class KalmanTrajectoryFilter:
    """
    Kalman Filter for trajectory prediction with uncertainty estimation.
    """
    
    def __init__(self, q_var=0.01, r_var=0.1):
        """
        Initialize the Kalman Filter.
        
        Args:
            q_var: Process noise variance
            r_var: Measurement noise variance
        """
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (we observe x and y positions)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * q_var
        
        # Measurement noise covariance
        self.R = np.eye(2) * r_var
        
        # Initial state and covariance
        self.x = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4)
        
    def predict(self):
        """
        Predict the next state.
        
        Returns:
            predicted_state: The predicted state [x, y, vx, vy]
            uncertainty: The uncertainty of the prediction (covariance)
        """
        # Predict the state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Return predicted position and its uncertainty
        predicted_position = self.x[:2]
        position_uncertainty = self.P[:2, :2]
        
        return predicted_position, position_uncertainty
    
    def update(self, measurement):
        """
        Update the state estimate with a new measurement.
        
        Args:
            measurement: The observed position [x, y]
        """
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def reset(self, initial_state):
        """
        Reset the filter with a new initial state.
        
        Args:
            initial_state: The initial state [x, y, vx, vy]
        """
        self.x = initial_state
        self.P = np.eye(4)
    
    def get_state_with_uncertainty(self):
        """
        Get the current state and its uncertainty.
        
        Returns:
            state: The current state [x, y, vx, vy]
            uncertainty: The uncertainty of the state (covariance)
        """
        return self.x, self.P
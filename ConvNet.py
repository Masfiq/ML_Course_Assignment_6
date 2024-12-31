# Replace next line with    class ConvNet(torch.nn.Module)
#   and continue making other changes as specified above.
class ConvNet(torch.nn.Module):  
    
    
    def __init__(self, input_shape, conv_specs, fc_specs, n_outputs, activation_function='tanh', device='cpu'):
        '''
        Example for MNIST (2D): ConvNet((1, 28, 28), [(20, 3, 1), (10, 4, 2)], [20], 10, 'tanh', 'cuda')
        Example for 1D data: ConvNet((1, 100), [(16, 3, 1), (32, 4, 2)], [64], 5, 'relu', 'cpu')
        '''
        super().__init__()

        self.input_shape = input_shape
        self.device = device
        print('ConvNet: Using device', self.device)

        # Select activation function
        self.activation_function = torch.nn.Tanh() if activation_function == 'tanh' else torch.nn.ReLU()

        # Determine whether input is 1D or 2D based on input_shape
        self.is_1d = len(input_shape) == 2  # 1D if input_shape = (C, L); 2D if input_shape = (C, H, W)

        # Create all convolutional layers
        n_in = input_shape[0]  # Number of input channels
        self.conv_layers = torch.nn.Sequential()
        for nh, patch_size, stride in conv_specs:
            if self.is_1d:
                self.conv_layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(n_in, nh, patch_size, stride),
                        self.activation_function
                    )
                )
            else:
                self.conv_layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(n_in, nh, patch_size, stride),
                        self.activation_function
                    )
                )
            n_in = nh

        # Pass zero input to determine the size of the flattened input to the fully connected layers
        zero_input = torch.zeros([1] + list(input_shape)).to(device)  # Example input for size calculation
        z = self.conv_layers(zero_input)
        z = z.reshape(1, -1)
        n_in = z.shape[1]

        # Create all fully connected layers
        self.fc_layers = torch.nn.Sequential()
        for nh in fc_specs:
            self.fc_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_in, nh),
                    self.activation_function
                )
            )
            n_in = nh
    
        # Add the output layer
        output_layer = torch.nn.Linear(n_in, n_outputs)
        self.fc_layers.append(output_layer)
    
        # Additional attributes for training
        self.pc_trace = []
        self.best_pc_val = None
    
        # Move model to the specified device
        self.to(self.device)
        
    def _forward_all_outputs(self, X):
        """
        Compute the outputs of all layers (both convolutional and fully connected) for the given input.
        Supports both 1D and 2D input data.
        """
        n_samples = X.shape[0]
        Zs = [X]
        
        # Pass input through all convolutional layers
        for conv_layer in self.conv_layers:
            Zs.append(conv_layer(Zs[-1]))
        
            # Flatten outputs from the last convolutional layer
        if self.is_1d:
            Zs[-1] = Zs[-1].reshape(n_samples, -1)  # For 1D: Flatten after Conv1d
        else:
            Zs[-1] = Zs[-1].reshape(n_samples, -1)  # For 2D: Flatten after Conv2d
        
            # Pass flattened data through fully connected layers
        for fc_layer in self.fc_layers:
           Zs.append(fc_layer(Zs[-1]))
        
        return Zs
               

    def forward(self, X, keep_all_outputs=False):
        """
        Forward pass through the network. Returns the output from the final layer.
        Optionally stores outputs from all layers if keep_all_outputs=True.
        
        Parameters:
        - X: Input data (1D or 2D tensor).
        - keep_all_outputs: Whether to store outputs from all layers.
        
        Returns:
        - Output from the final layer (logits for classification tasks).
        """
        # Convert input to a torch.Tensor if it isn't already
        if not isinstance(X, torch.Tensor):
            X = self._X_as_torch(X)
        
        # Forward pass through all layers
        Zs = self._forward_all_outputs(X)
        
        # Store outputs from all layers if required
        if keep_all_outputs:
            self.Zs = Zs  # Keep intermediate outputs for debugging or analysis
    
        # Return the output from the final layer
        return Zs[-1]

    
    def _X_as_torch(self, X):
        """
        Converts input data (X) to a PyTorch tensor.
        Reshapes it according to the model's input shape and moves it to the specified device.
        
        Parameters:
        - X: Input data, either a NumPy array or a PyTorch tensor.
    
        Returns:
        - PyTorch tensor version of X, reshaped and moved to the correct device.
        """
        if isinstance(X, torch.Tensor):
            return X
        else:
        # Reshape to match the expected input shape
            return torch.from_numpy(X.reshape([-1] + list(self.input_shape)).astype(np.float32)).to(self.device)


    def _T_as_torch(self, T):
        """
        Converts target labels (T) to a PyTorch tensor.
        Moves the tensor to the specified device.
        
        Parameters:
        - T: Target labels, either a NumPy array or a PyTorch tensor.
    
        Returns:
        - PyTorch tensor version of T, moved to the correct device.
        """
        if isinstance(T, torch.Tensor):
            return T
        else:
            return torch.from_numpy(T.astype(np.int64)).to(self.device)

    def percent_correct(self, Yclasses, T):
        """
        Computes the percentage of correctly classified samples.
        
        Parameters:
        - Yclasses: Predicted classes (tensor or array).
        - T: True target labels (tensor or array).
    
        Returns:
        - Percentage of correctly classified samples (float).
        """
        if isinstance(T, torch.Tensor):
            return (Yclasses == T).float().mean().item() * 100
        else:
            return (Yclasses == T).mean().item() * 100

    
    def train(self, Xtrain, Ttrain, Xval, Tval, n_epochs, batch_size=-1, method='sgd', learning_rate=0.01, verbose=True):
        """
        Train the ConvNet model on the provided training data.
        
        Parameters:
        - Xtrain: Training inputs (NumPy array or Tensor).
        - Ttrain: Training targets (NumPy array or Tensor).
        - Xval: Validation inputs (NumPy array or Tensor).
        - Tval: Validation targets (NumPy array or Tensor).
        - n_epochs: Number of training epochs.
        - batch_size: Batch size (default: -1, meaning full batch).
        - method: Optimization method ('sgd' or 'adam').
        - learning_rate: Learning rate for the optimizer.
        - verbose: Print training progress if True.
        
        Returns:
        - self: The trained model.
        """
        # Determine the unique classes in Ttrain
        self.classes = np.unique(Ttrain)
    
        # Standardize the training and validation data
        self.X_means = Xtrain.mean(0)
        self.X_stds = Xtrain.std(0)
        self.X_stds[self.X_stds == 0] = np.mean(self.X_stds)  # Avoid division by zero
    
        Xtrain = (Xtrain - self.X_means) / self.X_stds
        Xval = (Xval - self.X_means) / self.X_stds
    
        # Select optimizer
        if method == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Define the loss function
        loss_f = torch.nn.CrossEntropyLoss()
    
        # Set batch size to the full dataset size if not specified
        if batch_size == -1:
            batch_size = Xtrain.shape[0]
    
        self.batch_size = batch_size  # Save batch size for reference

        # Training loop
        for epoch in range(n_epochs):
            for first in range(0, Xtrain.shape[0], batch_size):
                # Create mini-batches
                Xtrain_batch = Xtrain[first:first + batch_size]
                Ttrain_batch = Ttrain[first:first + batch_size]
    
                # Convert to torch tensors
                Xtrain_batch = self._X_as_torch(Xtrain_batch)
                Ttrain_batch = self._T_as_torch(Ttrain_batch)
    
                # Forward pass
                Y = self(Xtrain_batch)
                loss = loss_f(Y, Ttrain_batch)
    
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation and performance tracking
            with torch.no_grad():
                pc_train = self.percent_correct(self.use(Xtrain, standardized=True), Ttrain)
                pc_val = self.percent_correct(self.use(Xval, standardized=True), Tval)
                self.pc_trace.append([pc_train, pc_val])
    
                # Track the best validation accuracy
                if self.best_pc_val is None or pc_val > self.best_pc_val:
                    self.best_pc_val = pc_val
                    self.best_epoch = epoch + 1
                    # Save the best model parameters
                    self.best_parameters = [p.clone() for p in self.parameters()]

            # Verbose output for progress monitoring
            if verbose and (epoch + 1) % max(1, (n_epochs // 10)) == 0:
                print(f'{method} Epoch {epoch + 1} % Correct: Train {self.pc_trace[-1][0]:.1f}'
                      f' Val {self.pc_trace[-1][1]:.1f}')

        # Restore the best parameters after training
        for p, bestp in zip(self.parameters(), self.best_parameters):
            p.data = bestp.clone()
    
        return self


    def _softmax(self, Y):
        """
        Compute the softmax of the given logits Y.
        A trick is used to avoid numerical overflow by subtracting the max value in each row.
    
        Parameters:
        - Y: Logits (Tensor) of shape (batch_size, n_classes).
    
        Returns:
        - Softmax probabilities (Tensor) of shape (batch_size, n_classes).
        """
        # Trick to avoid overflow: subtract max value along each row
        maxY = torch.max(Y, axis=1, keepdim=True)[0]  # Max per row (keep dimension for broadcasting)
        expY = torch.exp(Y - maxY)  # Exponentiate shifted logits
        denom = torch.sum(expY, axis=1, keepdim=True)  # Sum of exponentials per row
        Y = expY / denom  # Normalize to probabilities
        return Y


    def use(self, X, standardized=False, return_probs=False, keep_all_outputs=False):
        """
        Use the trained ConvNet model to predict class labels (and optionally probabilities) for input data.
    
        Parameters:
        - X: Input data (NumPy array or Tensor).
        - standardized: If False, standardize the input using stored means and standard deviations.
        - return_probs: If True, also return the predicted probabilities.
        - keep_all_outputs: If True, keep outputs from all layers during the forward pass.
    
        Returns:
        - classes: Predicted class labels (NumPy array or Tensor).
        - probs: (Optional) Predicted class probabilities (NumPy array or Tensor).
            """
        if not standardized:
            # Standardize input data using stored means and standard deviations
            X = (X - self.X_means) / self.X_stds
    
        classes = []  # To store predicted class labels
        probs = []    # To store predicted probabilities (if return_probs=True)
        return_numpy = False  # Flag to determine output type

        # Process data in batches
        for first in range(0, X.shape[0], self.batch_size):
            X_batch = X[first:first + self.batch_size]
    
            # Convert to PyTorch tensor if not already
            if not isinstance(X_batch, torch.Tensor):
                X_batch = self._X_as_torch(X_batch)
                return_numpy = True  # Track that input was originally NumPy
    
            with torch.no_grad():
                # Forward pass through the model
                Y = self(X_batch, keep_all_outputs=keep_all_outputs)
    
                # Predicted class indices
                class_index = torch.argmax(Y, axis=1).cpu().numpy()
                classes.extend(self.classes[class_index])
    
                # Compute probabilities if requested
                if return_probs:
                    probs.extend(self._softmax(Y).cpu().numpy())

        # Convert results to NumPy arrays if needed
        if return_numpy:
            classes = np.array(classes)
            probs = np.array(probs)

        # Return both classes and probabilities if requested
        if return_probs:
            return classes, probs
        else:
            return classes
import flwr as fl
# from flwr.server.strategy import FedAvg

import random
random.seed(10)

from variables import filename


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    
    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate evaluation losses using weighted average."""
        
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses     = [r.metrics["loss"] * r.num_examples for _, r in results]
        
        examples   = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        
        loss_aggregated = sum(losses) / sum(examples)
        print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
        
        
        with open(filename, 'a') as text_file:
            text_file.write("{},{},{}\n".format(rnd, accuracy_aggregated, loss_aggregated))

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    
    strategy = AggregateCustomMetricStrategy(
                      min_available_clients = num_clients,
                      fraction_fit          = fraction_fit,
                      min_fit_clients       = num_clients
                     )
    
    config   = {"num_rounds": num_rounds}
    
    # Exposes the server by default on port 8080
    fl.server.start_server(strategy = strategy, config = config)
    
    
    
    

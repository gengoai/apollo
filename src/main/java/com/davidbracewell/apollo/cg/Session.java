package com.davidbracewell.apollo.cg;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class Session {

   private Map<Placeholder, NDArray> defined = new HashMap<>();

   public Session define(@NonNull Placeholder p, double value) {
      defined.put(p, NDArrayFactory.DENSE_DOUBLE.scalar(value));
      return this;
   }

   public Session define(@NonNull Placeholder p, @NonNull NDArray value) {
      defined.put(p, value);
      return this;
   }

   private List<ComputationNode> postorder(ComputationNode node) {
      List<ComputationNode> toReturn = new ArrayList<>();

      if (node instanceof Operation) {
         Operation op = Cast.as(node);
         for (ComputationNode computationNode : op.getInputNodes()) {
            toReturn.addAll(postorder(computationNode));
         }
      }
      toReturn.add(node);
      return toReturn;
   }

   public NDArray run(Operation operation) {
      List<ComputationNode> nodes = postorder(operation);

      for (ComputationNode node : nodes) {
         if (node instanceof Operation) {
            ((Operation) node).compute();
         } else if (node instanceof Placeholder) {
            node.setOutput(defined.get(node));
         }
      }

      return operation.getOutput();
   }

}// END OF Session

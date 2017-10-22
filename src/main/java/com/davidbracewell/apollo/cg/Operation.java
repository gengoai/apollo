package com.davidbracewell.apollo.cg;

import com.davidbracewell.apollo.linear.NDArray;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public abstract class Operation implements ComputationNode {
   private final ComputationNode[] inputNodes;
   private NDArray output;

   public Operation() {
      this.inputNodes = new ComputationNode[0];
   }

   public Operation(@NonNull ComputationNode[] nodes) {
      this.inputNodes = nodes;
   }

   public abstract void compute();

   public ComputationNode[] getInputNodes() {
      return inputNodes;
   }

   @Override
   public NDArray getOutput() {
      return output;
   }

   @Override
   public void setOutput(NDArray output) {
      this.output = output;
   }
}// END OF Operation

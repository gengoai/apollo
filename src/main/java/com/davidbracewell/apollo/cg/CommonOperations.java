package com.davidbracewell.apollo.cg;

import com.davidbracewell.apollo.linear.NDArray;
import lombok.NonNull;
import lombok.val;

import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;

/**
 * @author David B. Bracewell
 */
public final class CommonOperations {

   private CommonOperations() {
      throw new IllegalAccessError();
   }

   public static Operation add(@NonNull ComputationNode x, @NonNull ComputationNode y) {
      return new Add(new ComputationNode[]{x, y});
   }

   public static Operation dot(@NonNull ComputationNode x, @NonNull ComputationNode y) {
      return new Dot(new ComputationNode[]{x, y});
   }

   public static Operation map(@NonNull ComputationNode x, @NonNull ComputationNode y, @NonNull DoubleBinaryOperator operator) {
      return new BinaryOperation((n1, n2) -> n1.map(n2, operator), x, y);
   }

   public static Operation mmul(@NonNull ComputationNode x, @NonNull ComputationNode y) {
      return new MatrixMul(new ComputationNode[]{x, y});
   }

   public static Operation sub(@NonNull ComputationNode x, @NonNull ComputationNode y) {
      return new Subtract(new ComputationNode[]{x, y});
   }

   private static class BinaryOperation extends Operation {

      private final BinaryOperator<NDArray> operator;

      private BinaryOperation(BinaryOperator<NDArray> operator, ComputationNode a, ComputationNode b) {
         super(new ComputationNode[]{a, b});
         this.operator = operator;
      }

      @Override
      public void compute() {
         setOutput(operator.apply(getInputNodes()[0].getOutput(), getInputNodes()[1].getOutput()));
      }
   }

   private static class Dot extends Operation {

      public Dot(ComputationNode[] nodes) {
         super(nodes);
      }

      @Override
      public void compute() {
         setOutput(getInputNodes()[0].getOutput().dot(getInputNodes()[1].getOutput()));
      }
   }

   private static class MatrixMul extends Operation {

      public MatrixMul(ComputationNode[] nodes) {
         super(nodes);
      }

      @Override
      public void compute() {
         setOutput(getInputNodes()[0].getOutput().mmul(getInputNodes()[1].getOutput()));
      }
   }

   private static class Subtract extends Operation {

      public Subtract(ComputationNode[] nodes) {
         super(nodes);
      }

      @Override
      public void compute() {
         setOutput(getInputNodes()[0].getOutput().sub(getInputNodes()[1].getOutput()));
      }
   }

   private static class Add extends Operation {

      public Add(ComputationNode[] nodes) {
         super(nodes);
      }

      @Override
      public void compute() {
         val x = getInputNodes()[0].getOutput();
         val y = getInputNodes()[1].getOutput();
         if (x.isScalar()) {
            setOutput(y.add(x.get(0)));
         } else if (y.isScalar()) {
            setOutput(x.add(y.get(0)));
         } else {
            setOutput(x.add(y));
         }
      }
   }

}// END OF CommonOperations

package com.davidbracewell.apollo.cg;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class ComputationGraph implements Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private final List<Operation> operations = new ArrayList<>();
   @Getter
   private final List<Variable> variables = new ArrayList<>();
   @Getter
   private final List<Placeholder> placeholders = new ArrayList<>();


   public static void main(String[] args) {
      ComputationGraph graph = new ComputationGraph();
      val A = graph.createVariable(NDArrayFactory.DENSE_DOUBLE.rand(3));
      val B = graph.createVariable(NDArrayFactory.DENSE_DOUBLE.scalar(100));
      val C = graph.createPlaceholder();
      val z = graph.add(CommonOperations.add(A, B));
      val session = new Session()
                       .define(C, NDArrayFactory.DENSE_DOUBLE.rand(3));
      System.out.println(session.run(z));
   }

   public void add(ComputationNode... nodes) {
      for (ComputationNode node : nodes) {
         if (node instanceof Operation) {
            operations.add(Cast.as(node));
         } else if (node instanceof Variable) {
            variables.add(Cast.as(node));
         } else {
            placeholders.add(Cast.as(node));
         }
      }
   }

   public Operation add(@NonNull Operation operation) {
      this.operations.add(operation);
      return operation;
   }

   public void add(@NonNull Placeholder placeholder) {
      this.placeholders.add(placeholder);
   }

   public Variable add(@NonNull Variable variable) {
      this.variables.add(variable);
      return variable;
   }

   public Placeholder createPlaceholder() {
      Placeholder p = new Placeholder();
      add(p);
      return p;
   }

   public Variable createVariable(NDArray value) {
      Variable v = new Variable(value);
      add(v);
      return v;
   }

   public Variable createVariable(double value) {
      Variable v = new Variable(value);
      add(v);
      return v;
   }

   public Variable createVariable() {
      Variable v = new Variable();
      add(v);
      return v;
   }

}// END OF ComputationGraph

package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * <p>Associates a label with a vector.</p>
 *
 * @author David B. Bracewell
 */
public class LabeledVector extends ForwardingVector {
   private static final long serialVersionUID = 1L;
   private final Vector delegate;
   private Object label;

   /**
    * Instantiates a new Labeled vector.
    *
    * @param label    the label
    * @param delegate the delegate
    */
   public LabeledVector(Object label, @NonNull Vector delegate) {
      this.label = label;
      this.delegate = delegate;
   }

   /**
    * Convenience method for creating a sparse labeled vector.
    *
    * @param dimension the dimension of the vector
    * @param label     the label of the vector
    * @return the labeled vector
    */
   public static LabeledVector sparse(int dimension, Object label) {
      return new LabeledVector(label, new SparseVector(dimension));
   }

   /**
    * Convenience method for creating a dense labeled vector.
    *
    * @param dimension the dimension of the vector
    * @param label     the label of the vector
    * @return the labeled vector
    */
   public static LabeledVector dense(int dimension, Object label) {
      return new LabeledVector(label, new DenseVector(dimension));
   }

   @Override
   protected Vector delegate() {
      return delegate;
   }

   @Override
   public <T> T getLabel() {
      return Cast.as(label);
   }

   /**
    * Sets the label of the vector
    *
    * @param label the new label of the vector
    * @return This vector
    */
   public LabeledVector setLabel(Object label) {
      this.label = label;
      return this;
   }

   /**
    * Checks for a non-null label
    *
    * @return True if the label is non-null, False if null
    */
   public boolean hasLabel() {
      return label != null;
   }

   @Override
   public LabeledVector addSelf(Vector rhs) {
      super.addSelf(rhs);
      return this;
   }

   @Override
   public LabeledVector compress() {
      super.compress();
      return this;
   }

   @Override
   public LabeledVector copy() {
      return new LabeledVector(label, super.copy());
   }

   @Override
   public LabeledVector decrement(int index) {
      super.decrement(index);
      return this;
   }

   @Override
   public LabeledVector decrement(int index, double amount) {
      super.decrement(index, amount);
      return this;
   }

   @Override
   public LabeledVector divideSelf(Vector rhs) {
      super.divideSelf(rhs);
      return this;
   }

   @Override
   public LabeledVector increment(int index) {
      super.increment(index);
      return this;
   }

   @Override
   public LabeledVector increment(int index, double amount) {
      super.increment(index, amount);
      return this;
   }

   @Override
   public LabeledVector mapAddSelf(double amount) {
      super.mapAddSelf(amount);
      return this;
   }

   @Override
   public LabeledVector mapMultiplySelf(double amount) {
      super.mapMultiplySelf(amount);
      return this;
   }

   @Override
   public LabeledVector mapDivideSelf(double amount) {
      super.mapDivideSelf(amount);
      return this;
   }

   @Override
   public LabeledVector mapSelf(Vector v, DoubleBinaryOperator function) {
      super.mapSelf(v, function);
      return this;
   }

   @Override
   public LabeledVector mapSelf(DoubleUnaryOperator function) {
      super.mapSelf(function);
      return this;
   }

   @Override
   public LabeledVector mapSubtractSelf(double amount) {
      super.mapSubtractSelf(amount);
      return this;
   }

   @Override
   public LabeledVector multiplySelf(Vector rhs) {
      super.multiplySelf(rhs);
      return this;
   }

   @Override
   public LabeledVector scale(int index, double amount) {
      super.scale(index, amount);
      return this;
   }

   @Override
   public LabeledVector set(int index, double value) {
      super.set(index, value);
      return this;
   }

   @Override
   public LabeledVector subtractSelf(Vector rhs) {
      super.subtractSelf(rhs);
      return this;
   }

   @Override
   public LabeledVector redim(int newDimension) {
      return new LabeledVector(label, super.redim(newDimension));
   }

}// END OF LabeledVector

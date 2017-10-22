package com.davidbracewell.apollo.linear;

import com.davidbracewell.Copyable;
import lombok.EqualsAndHashCode;

/**
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = true)
public class Shape extends Subscript implements Copyable<Shape> {


   /**
    * Instantiates a new Shape object given two subscripts in a NDArray
    *
    * @param r the first dimension subscript
    * @param c the second dimension subscript
    */
   public Shape(int r, int c) {
      super(r, c);
   }

   public static Shape shape(int numberOfRows, int numberOfColumns) {
      return new Shape(numberOfRows, numberOfColumns);
   }

   public boolean canMultiply(Shape other) {
      return get(Axis.COlUMN) == other.get(Axis.ROW);
   }

   public void checkCanMultiply(Shape other) {
      if (!canMultiply(other)) {
         throw new IllegalArgumentException("Multiplication dimension mismatch: "
                                               + get(Axis.COlUMN)
                                               + " is not equal to "
                                               + other.get(Axis.ROW));
      }
   }

   public void checkDimensionMatch(Shape other) {
      if (!dimensionMatch(other)) {
         throw new IllegalArgumentException("Dimension mismatch: " + this + " is not equal to " + other);
      }
   }

   public void checkLength(Shape other){
      if (length() != other.length()){
         throw new IllegalArgumentException("Dimension mismatch " + length() + " != " + other.length());
      }
   }

   public void checkOppDimensionMatch(Shape other, Axis axis) {
      if (!oppDimensionMatch(other, axis)) {
         throw new IllegalArgumentException("Dimension mismatch: "
                                               + get(axis.T())
                                               + " is not equal to "
                                               + other.get(axis));
      }
   }

   public void checkDimensionMatch(Shape other, Axis axis) {
      if (!dimensionMatch(other, axis)) {
         throw new IllegalArgumentException("Dimension mismatch: "
                                               + get(axis)
                                               + " is not equal to "
                                               + other.get(axis));
      }
   }

   public void checkDimensionMatch(Axis tAxis, Shape other, Axis oAxis) {
      if (get(tAxis) != other.get(oAxis)) {
         throw new IllegalArgumentException("Dimension mismatch: "
                                               + get(tAxis)
                                               + " is not equal to "
                                               + other.get(oAxis));
      }
   }

   public void checkSubscript(Subscript ss) {
      if (ss.i < 0 || ss.i > this.i || ss.j < 0 || ss.j > this.j) {
         throw new IllegalArgumentException("Subscript " + ss + " out of range " + this);
      }
   }

   public int colMajorIndex(int i, int j) {
      return i + this.i * j;
   }

   public Subscript fromColMajorIndex(int index) {
      int r = index % this.i;
      int c = index / this.i;
      return Subscript.from(r, c);
   }


   public int colMajorIndex(Subscript s) {
      return colMajorIndex(s.i, s.j);
   }

   @Override
   public Shape copy() {
      return new Shape(i, j);
   }

   public boolean dimensionMatch(Shape other, Axis axis) {
      return get(axis) == other.get(axis);
   }

   public boolean oppDimensionMatch(Shape other, Axis axis) {
      return get(axis.T()) == other.get(axis);
   }


   public boolean dimensionMatch(Shape other) {
      return this.equals(other);
   }

   public Subscript fromRowMajorIndex(int index) {
      int r = index / this.j;
      int c = index % this.j;
      return Subscript.from(r, c);
   }

   public int length() {
      return i * j;
   }

   public int rowMajorIndex(int i, int j) {
      return i * this.j + j;
   }

   public int rowMajorIndex(Subscript s) {
      return rowMajorIndex(s.i, s.j);
   }

   @Override
   public String toString() {
      return "(" + i + ", " + j + ")";
   }

}// END OF Shape

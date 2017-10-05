package com.davidbracewell.apollo.linear;

import lombok.EqualsAndHashCode;
import lombok.NonNull;

import java.io.Serializable;

/**
 * An indexed position in a NDArray. Given an NDArray <code>A</code> that is of shape <code>(10,10)</code> an example
 * subscript would be <code>(5,4)</code> referring to the fifth row and fourth column.
 *
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
public class Subscript implements Serializable, Comparable<Subscript> {
   /**
    * The subscript of the first dimension (i.e. row in a matrix)
    */
   public final int i;
   /**
    * The subscript of the second dimension (i.e. column in a matrix)
    */
   public final int j;

   /**
    * Instantiates a new Subscript object given two subscripts in a NDArray
    *
    * @param i the first dimension subscript
    * @param j the second dimension subscript
    */
   protected Subscript(int i, int j) {
      this.i = i;
      this.j = j;
   }

   /**
    * Creates a Subscript object from the given r and c subscripts
    *
    * @param r the row subscript
    * @param c the column subscript
    * @return the subscript
    */
   public static Subscript from(int r, int c) {
      return new Subscript(r, c);
   }


   @Override
   public int compareTo(@NonNull Subscript o) {
      int cmp = Integer.compare(i, o.i);
      if (cmp == 0) {
         return Integer.compare(j, o.j);
      }
      return cmp;
   }

   /**
    * Gets the subscript of the dimension for the given axis
    *
    * @param axis the axis whose subscript we want
    * @return the subscript for the given axis
    */
   public int get(@NonNull Axis axis) {
      return axis == Axis.ROW ? i : j;
   }


   @Override
   public String toString() {
      return "(" + i + ", " + j + ")";
   }
}// END OF Subscript

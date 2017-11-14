package com.davidbracewell.apollo.linear;

/**
 * Defines the axes that an NDArray can have
 *
 * @author David B. Bracewell
 */
public enum Axis {
   /**
    * Row axis.
    */
   ROW(0) {
      @Override
      public Axis T() {
         return COlUMN;
      }

      @Override
      public int select(int i, int j) {
         return i;
      }
   },
   /**
    * Column axis.
    */
   COlUMN(1) {
      @Override
      public Axis T() {
         return ROW;
      }

      @Override
      public int select(int i, int j) {
         return j;
      }
   };

   /**
    * Ordinal index (0 for row, 1 for Column).
    */
   final int index;

   Axis(int index) {
      this.index = index;
   }

   /**
    * Gets an axis object from its index (0 for row, 1 for column)
    *
    * @param index the ordinal value
    * @return the axis
    */
   public static Axis from(int index) {
      switch (index) {
         case 0:
            return ROW;
         case 1:
            return COlUMN;
      }
      throw new IllegalArgumentException("Dimension index (" + index + ") does not map to a known axis.");
   }

   /**
    * Gets this axis's opposite, or transposed, axis
    *
    * @return the axis
    */
   public abstract Axis T();

   /**
    * Selects dimension value associated with this axis
    *
    * @param i dimension one
    * @param j dimension two
    * @return the dimension associated with this axis
    */
   public abstract int select(int i, int j);


}// END OF Axis

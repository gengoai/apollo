package com.davidbracewell.apollo.linear;

/**
 * The enum Axis.
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
    * C ol umn axis.
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
    * The Index.
    */
   final int index;

   Axis(int index) {
      this.index = index;
   }

   public abstract Axis T();

   public abstract int select(int i, int j);

   /**
    * From axis.
    *
    * @param index the index
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


}// END OF Axis

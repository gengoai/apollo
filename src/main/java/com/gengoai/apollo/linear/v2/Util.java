package com.gengoai.apollo.linear.v2;

/**
 * @author David B. Bracewell
 */
final class Util {
   private Util() {
      throw new IllegalAccessError();
   }


   static int sliceIndex(int kernel, int channel, int numChannel) {
      return kernel + (numChannel * channel);
   }

   static int[] reverseSliceIndex(int index, int numKernel) {
      return new int[]{index % numKernel, index / numKernel};
   }

   static int rowColumn(int kernel, int channel, int numChannel) {
      return kernel + (numChannel * channel);
   }

   static int[] reverseRowColumnIndex(int index, int numKernel) {
      return new int[]{index % numKernel, index / numKernel};
   }

   static int selectRowCol(int axis, int r, int c) {
      return axis == NDArray.ROW ? r : c;
   }


   static int oppositeIndex(int index) {
      switch (index) {
         case NDArray.ROW:
            return 1;
         case NDArray.COLUMN:
            return 0;
         case NDArray.KERNEL:
            return 3;
         case NDArray.CHANNEL:
            return 2;
      }
      throw new IllegalArgumentException();
   }

}//END OF Util

package com.davidbracewell.apollo.linalg;

/**
 * @author David B. Bracewell
 */
class ColumnVector extends BaseVector {
   private static final long serialVersionUID = 1L;
   final int column;
   private Matrix entries;

   ColumnVector(Matrix entries, int column) {
      this.entries = entries;
      this.column = column;
   }

   @Override
   public Vector compress() {
      return this;
   }

   @Override
   public Vector copy() {
      return new DenseVector(toArray())
                .setLabel(getLabel())
                .setWeight(getWeight())
                .setPredicted(getPredicted());
   }

   @Override
   protected Vector createNew(int dimension) {
      return entries.isSparse() ? Vector.sZeros(dimension) : Vector.dZeros(dimension);
   }

   @Override
   public int dimension() {
      return entries.numberOfRows();
   }

   @Override
   public double get(int index) {
      return entries.get(index, column);
   }

   @Override
   public Vector increment(int index, double amount) {
      entries.increment(index, column, amount);
      return this;
   }

   @Override
   public boolean isDense() {
      return true;
   }

   @Override
   public boolean isSparse() {
      return false;
   }


   @Override
   public Vector set(int index, double value) {
      entries.set(index, column, value);
      return this;
   }

   @Override
   public int size() {
      return entries.numberOfRows();
   }

   @Override
   public Vector zero() {
      for (int r = 0; r < entries.numberOfRows(); r++) {
         set(r, 0d);
      }
      return this;
   }
}

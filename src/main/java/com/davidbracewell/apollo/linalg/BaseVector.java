package com.davidbracewell.apollo.linalg;

import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.apollo.affinity.Correlation;
import com.davidbracewell.apollo.optimization.Optimum;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * The type Base vector.
 *
 * @author David B. Bracewell
 */
public abstract class BaseVector implements Vector, Serializable {
   private static final long serialVersionUID = 1L;
   private Object label;
   private double weight;
   private double predicted;


   @Override
   public Vector add(@NonNull Vector rhs) {
      return copy().addSelf(rhs);
   }

   @Override
   public Vector addSelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      rhs.forEachSparse(e -> increment(e.index, e.value));
      return this;
   }

   @Override
   public Map<Integer, Double> asMap() {
      return VectorMap.wrap(this);
   }

   @Override
   public Vector compress() {
      return this;
   }

   @Override
   public Vector concat(@NonNull Vector other) {
      Vector vPrime = createNew(dimension() + other.dimension());
      nonZeroIterator().forEachRemaining(e -> vPrime.set(e.index, e.value));
      other.nonZeroIterator().forEachRemaining(e -> vPrime.set(e.index, e.value));
      return vPrime;
   }

   @Override
   public Vector copy() {
      Vector vPrime = createNew(dimension());
      forEach(e -> vPrime.set(e.index, e.value));
      return vPrime;
   }

   @Override
   public double corr(@NonNull Vector other) {
      return Correlation.Spearman.calculate(this, other);
   }

   /**
    * Create new vector.
    *
    * @param dimension the dimension
    * @return the vector
    */
   protected abstract Vector createNew(int dimension);

   @Override
   public Vector decrement(int index) {
      return increment(index, -1);
   }

   @Override
   public Vector decrement(int index, double amount) {
      return increment(index, -amount);
   }

   @Override
   public Vector divide(@NonNull Vector rhs) {
      return copy().divideSelf(rhs);
   }

   @Override
   public Vector divideSelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      forEachSparse(e -> scale(e.index, 1d / rhs.get(e.index)));
      return this;
   }

   @Override
   public double dot(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      AtomicDouble dot = new AtomicDouble(0d);
      Vector small = size() < rhs.size() ? this : rhs;
      Vector big = size() < rhs.size() ? rhs : this;
      small.forEachSparse(e -> dot.addAndGet(e.value * big.get(e.index)));
      return dot.get();
   }

   @Override
   public void forEachOrderedSparse(@NonNull Consumer<Entry> consumer) {
      Streams.asStream(orderedNonZeroIterator()).forEach(consumer);
   }

   @Override
   public void forEachSparse(@NonNull Consumer<Vector.Entry> consumer) {
      Streams.asStream(nonZeroIterator()).forEach(consumer);
   }

   @Override
   public <T> T getLabel() {
      return Cast.as(label);
   }

   @Override
   public double getLabelAsDouble() {
      if (label == null || !(label instanceof Number)) {
         return Double.NaN;
      }
      return Cast.<Number>as(label).doubleValue();
   }

   @Override
   public double getPredicted() {
      return predicted;
   }

   @Override
   public double getWeight() {
      return weight;
   }

   @Override
   public Vector increment(int index) {
      return increment(index, 1);
   }

   @Override
   public Vector insert(int i, double v) {
      Preconditions.checkArgument(i >= 0, "insert location must be >= 0");
      Vector vPrime;
      if (i <= dimension()) {
         vPrime = createNew(dimension() + 1);
      } else {
         vPrime = createNew(dimension() + (i - dimension() + 1));
      }
      vPrime.set(i, v);
      for (int j = 0; j < i && j < dimension(); j++) {
         vPrime.set(j, get(j));
      }
      vPrime.set(i, v);
      for (int j = i + 1; j < dimension(); j++) {
         vPrime.set(j + 1, get(j));
      }
      return vPrime;
   }

   @Override
   public boolean isDense() {
      return false;
   }

   @Override
   public boolean isFinite() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).allMatch(Double::isFinite);
   }

   @Override
   public boolean isInfinite() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).anyMatch(Double::isInfinite);
   }

   @Override
   public boolean isNaN() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).anyMatch(Double::isNaN);
   }

   @Override
   public boolean isSparse() {
      return false;
   }

   @Override
   public Iterator<Entry> iterator() {
      return new Iterator<Entry>() {
         private final PrimitiveIterator.OfInt indexIter = IntStream.range(0, dimension()).iterator();

         @Override
         public boolean hasNext() {
            return indexIter.hasNext();
         }

         @Override
         public Entry next() {
            if (!indexIter.hasNext()) {
               throw new NoSuchElementException();
            }
            int index = indexIter.next();
            return new Vector.Entry(index, get(index));
         }
      };
   }

   @Override
   public double l1Norm() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).map(Math::abs).sum();
   }

   @Override
   public double lInfNorm() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).map(Math::abs).max().orElse(0d);
   }

   @Override
   public double magnitude() {
      return Math.sqrt(Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).map(d -> d * d).sum());
   }

   @Override
   public Vector map(@NonNull DoubleUnaryOperator function) {
      return copy().mapSelf(function);
   }

   @Override
   public Vector map(@NonNull Vector v, @NonNull DoubleBinaryOperator function) {
      return copy().mapSelf(v, function);
   }

   @Override
   public Vector mapAdd(double amount) {
      return copy().mapAddSelf(amount);
   }

   @Override
   public Vector mapAddSelf(double amount) {
      for (int i = 0; i < dimension(); i++) {
         increment(i, amount);
      }
      return this;
   }

   @Override
   public Vector mapDivide(double amount) {
      return copy().mapDivideSelf(amount);
   }

   @Override
   public Vector mapDivideSelf(double amount) {
      forEachSparse(e -> scale(e.index, 1d / amount));
      return this;
   }

   @Override
   public Vector mapMultiply(double amount) {
      return copy().mapMultiplySelf(amount);
   }

   @Override
   public Vector mapMultiplySelf(double amount) {
      forEachSparse(e -> scale(e.index, amount));
      return this;
   }

   @Override
   public Vector mapSelf(@NonNull Vector v, @NonNull DoubleBinaryOperator function) {
      Preconditions.checkArgument(v.dimension() == dimension(), "Dimension mismatch");
      for (int i = 0; i < dimension(); i++) {
         set(i, function.applyAsDouble(get(i), v.get(i)));
      }
      return this;
   }

   @Override
   public Vector mapSelf(@NonNull DoubleUnaryOperator function) {
      for (int i = 0; i < dimension(); i++) {
         set(i, function.applyAsDouble(get(i)));
      }
      return this;
   }

   @Override
   public Vector mapSubtract(double amount) {
      return copy().mapSubtractSelf(amount);
   }

   @Override
   public Vector mapSubtractSelf(double amount) {
      for (int i = 0; i < dimension(); i++) {
         decrement(i, amount);
      }
      return this;
   }

   @Override
   public double max() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).max().orElse(0d);
   }

   @Override
   public int maxIndex() {
      return Optimum.MAXIMUM.optimumIndex(toArray());
   }

   @Override
   public double min() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).min().orElse(0d);
   }

   @Override
   public int minIndex() {
      return Optimum.MINIMUM.optimumIndex(toArray());
   }

   @Override
   public Vector multiply(@NonNull Vector rhs) {
      return copy().multiplySelf(rhs);
   }

   @Override
   public Vector multiplySelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      forEachSparse(e -> scale(e.index, rhs.get(e.index)));
      return this;
   }

   @Override
   public Iterator<Vector.Entry> nonZeroIterator() {
      return orderedNonZeroIterator();
   }

   @Override
   public Iterator<Vector.Entry> orderedNonZeroIterator() {
      return new Iterator<Entry>() {
         private final PrimitiveIterator.OfInt indexIter = IntStream.range(0, dimension()).iterator();
         private Integer ni = null;

         private boolean advance() {
            while (ni == null) {
               if (indexIter.hasNext()) {
                  ni = indexIter.next();
                  if (get(ni) == 0) {
                     ni = null;
                  }
               } else {
                  return false;
               }
            }
            return ni != null;
         }

         @Override
         public boolean hasNext() {
            return advance();
         }

         @Override
         public Entry next() {
            if (!advance()) {
               throw new NoSuchElementException();
            }
            int index = ni;
            ni = null;
            return new Vector.Entry(index, get(index));
         }
      };
   }

   @Override
   public Vector redim(int newDimension) {
      Preconditions.checkArgument(newDimension > 0, "Only positive dimensions allowed.");
      Vector vPrime = createNew(newDimension);
      for (int i = 0; i < Math.min(newDimension, dimension()); i++) {
         vPrime.set(i, get(i));
      }
      return vPrime;
   }

   @Override
   public Vector scale(int index, double amount) {
      Preconditions.checkPositionIndex(index, dimension());
      set(index, get(index) * amount);
      return this;
   }

   @Override
   public Vector setLabel(Object o) {
      this.label = o;
      return this;
   }

   @Override
   public Vector setPredicted(double predicted) {
      this.predicted = predicted;
      return this;
   }

   @Override
   public Vector setWeight(double weight) {
      this.weight = weight;
      return this;
   }

   @Override
   public Vector slice(int from, int to) {
      Preconditions.checkPositionIndex(from, dimension());
      Preconditions.checkPositionIndex(to, dimension() + 1);
      Preconditions.checkState(to > from, "To index must be > from index");
      Vector vPrime = createNew(to - from);
      for (int i = from; i < to; i++) {
         vPrime.set(i - from, get(i));
      }
      return vPrime;
   }

   @Override
   public Tuple2<int[], double[]> sparse() {
      int[] indices = new int[size()];
      double[] values = new double[size()];
      int i = 0;
      for (Entry entry : Collect.asIterable(orderedNonZeroIterator())) {
         indices[i] = entry.getIndex();
         values[i] = entry.getValue();
         i++;
      }
      return $(indices, values);
   }

   @Override
   public EnhancedDoubleStatistics statistics() {
      return DoubleStream.of(toArray()).collect(EnhancedDoubleStatistics::new, EnhancedDoubleStatistics::accept,
                                                EnhancedDoubleStatistics::combine);
   }

   @Override
   public Vector subtract(@NonNull Vector rhs) {
      return copy().subtractSelf(rhs);
   }

   @Override
   public Vector subtractSelf(@NonNull Vector rhs) {
      Preconditions.checkArgument(rhs.dimension() == dimension(), "Dimension mismatch");
      rhs.forEachSparse(e -> decrement(e.index, e.value));
      return this;
   }

   @Override
   public double sum() {
      return Streams.asStream(nonZeroIterator()).mapToDouble(Entry::getValue).sum();
   }

   @Override
   public double[] toArray() {
      double[] array = new double[dimension()];
      forEach(e -> array[e.getIndex()] = e.getValue());
      return array;
   }

   @Override
   public Matrix toDiagMatrix() {
      SparseMatrix matrix = new SparseMatrix(dimension(), dimension());
      for (int i = 0; i < dimension(); i++) {
         matrix.set(i, i, get(i));
      }
      return matrix;
   }

   @Override
   public Matrix toMatrix() {
      if (isSparse()) {
         return new SparseMatrix(this);
      }
      return new DenseMatrix(1, dimension(), this.toArray());
   }

   @Override
   public Vector toUnitVector() {
      return copy().mapDivideSelf(magnitude());
   }

   @Override
   public Matrix transpose() {
      Matrix matrix = isSparse() ? new SparseMatrix(dimension(), 1) :
                      new DenseMatrix(dimension(), 1);
      for (int i = 0; i < dimension(); i++) {
         matrix.set(i, 0, get(i));
      }
      return matrix;
   }

   @Override
   public Vector zero() {
      return createNew(dimension());
   }


}// END OF BaseVector

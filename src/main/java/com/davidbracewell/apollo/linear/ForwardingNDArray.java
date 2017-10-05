package com.davidbracewell.apollo.linear;

import com.davidbracewell.EnhancedDoubleStatistics;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.Serializable;
import java.util.Iterator;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

/**
 * @author David B. Bracewell
 */
public class ForwardingNDArray implements NDArray, Serializable {
   private static final long serialVersionUID = 1L;
   final NDArray delegate;

   public ForwardingNDArray(NDArray delegate) {
      this.delegate = delegate;
   }

   @Override
   public NDArray T() {
      return delegate.T();
   }

   @Override
   public NDArray add(double scalar) {
      return delegate.add(scalar);
   }

   @Override
   public NDArray add(NDArray other) {
      return delegate.add(other);
   }

   @Override
   public NDArray add(NDArray other, Axis axis) {
      return delegate.add(other, axis);
   }

   @Override
   public NDArray addi(double scalar) {
      delegate.addi(scalar);
      return this;
   }

   @Override
   public NDArray addi(NDArray other) {
      delegate.addi(other);
      return this;
   }

   @Override
   public NDArray addi(NDArray other, Axis axis) {
      delegate.addi(other, axis);
      return this;
   }

   @Override
   public int[] argMax(Axis axis) {
      return delegate.argMax(axis);
   }

   @Override
   public int[] argMin(Axis axis) {
      return delegate.argMin(axis);
   }

   @Override
   public NDArray compress() {
      delegate.compress();
      return this;
   }

   @Override
   public NDArray copy() {
      return delegate.copy();
   }

   @Override
   public NDArray decrement(int index) {
      delegate.decrement(index);
      return this;
   }

   @Override
   public NDArray decrement(int index, double amount) {
      delegate.decrement(index, amount);
      return this;
   }

   @Override
   public NDArray decrement(int i, int j) {
      delegate.decrement(i, j);
      return this;
   }

   @Override
   public NDArray decrement(int i, int j, double amount) {
      delegate.decrement(i, j, amount);
      return this;
   }

   protected NDArray delegate() {
      return delegate;
   }

   @Override
   public NDArray diag() {
      return delegate.diag();
   }

   @Override
   public NDArray div(double scalar) {
      return delegate.div(scalar);
   }

   @Override
   public NDArray div(NDArray other) {
      return delegate.div(other);
   }

   @Override
   public NDArray div(NDArray other, Axis axis) {
      return delegate.div(other, axis);
   }

   @Override
   public NDArray divi(double scalar) {
      delegate.divi(scalar);
      return this;
   }

   @Override
   public NDArray divi(NDArray other) {
      delegate.divi(other);
      return this;
   }

   @Override
   public NDArray divi(NDArray other, Axis axis) {
      delegate.divi(other, axis);
      return this;
   }

   @Override
   public double dot(NDArray other) {
      return delegate.dot(other);
   }

   @Override
   public NDArray exp() {
      return delegate.exp();
   }

   @Override
   public NDArray expi() {
      delegate.expi();
      return this;
   }

   @Override
   public NDArray fill(double value) {
      delegate.fill(value);
      return this;
   }

   @Override
   public void forEach(Consumer<Entry> consumer) {
      delegate.forEach(consumer);
   }

   @Override
   public void forEachSparse(Consumer<Entry> consumer) {
      delegate.forEachSparse(consumer);
   }

   @Override
   public void forEachSparseOrdered(Consumer<Entry> consumer) {
      delegate.forEachSparseOrdered(consumer);
   }

   @Override
   public double get(int index) {
      return delegate.get(index);
   }

   @Override
   public double get(int i, int j) {
      return delegate.get(i, j);
   }

   @Override
   public double get(Subscript subscript) {
      return delegate.get(subscript);
   }

   @Override
   public double get(Axis a1, int dim1, Axis a2, int dim2) {
      return delegate.get(a1, dim1, a2, dim2);
   }

   @Override
   public NDArrayFactory getFactory() {
      return delegate.getFactory();
   }

   @Override
   public NDArray getVector(int index, Axis axis) {
      return delegate.getVector(index, axis);
   }

   @Override
   public NDArray increment(int index) {
      delegate.increment(index);
      return this;
   }

   @Override
   public NDArray increment(int index, double amount) {
      delegate.increment(index, amount);
      return this;
   }

   @Override
   public NDArray increment(int i, int j) {
      delegate.increment(i, j);
      return this;
   }

   @Override
   public NDArray increment(int i, int j, double amount) {
      delegate.increment(i, j, amount);
      return this;
   }

   @Override
   public boolean isColumnVector() {
      return delegate.isColumnVector();
   }

   @Override
   public boolean isEmpty() {
      return delegate.isEmpty();
   }

   @Override
   public boolean isRowVector() {
      return delegate.isRowVector();
   }

   @Override
   public boolean isScalar() {
      return delegate.isScalar();
   }

   @Override
   public boolean isSparse() {
      return delegate.isSparse();
   }

   @Override
   public boolean isSquare() {
      return delegate.isSquare();
   }

   @Override
   public boolean isVector() {
      return delegate.isVector();
   }

   @Override
   public boolean isVector(Axis axis) {
      return delegate.isVector(axis);
   }

   @Override
   public Iterator<Entry> iterator() {
      return delegate.iterator();
   }

   @Override
   public int length() {
      return delegate.length();
   }

   @Override
   public NDArray log() {
      return delegate.log();
   }

   @Override
   public NDArray logi() {
      delegate.logi();
      return this;
   }

   @Override
   public NDArray map(DoubleUnaryOperator operator) {
      return delegate.map(operator);
   }

   @Override
   public NDArray map(NDArray vector, Axis axis, DoubleBinaryOperator operator) {
      return delegate.map(vector, axis, operator);
   }

   @Override
   public NDArray map(NDArray other, DoubleBinaryOperator operator) {
      return delegate.map(other, operator);
   }

   @Override
   public NDArray mapIf(DoublePredicate predicate, DoubleUnaryOperator operator) {
      return delegate.mapIf(predicate, operator);
   }

   @Override
   public NDArray mapSparse(NDArray vector, Axis axis, DoubleBinaryOperator operator) {
      return delegate.mapSparse(vector, axis, operator);
   }

   @Override
   public NDArray mapSparse(NDArray other, DoubleBinaryOperator operator) {
      return delegate.mapSparse(other, operator);
   }

   @Override
   public NDArray mapSparse(DoubleUnaryOperator operator) {
      return delegate.mapSparse(operator);
   }

   @Override
   public NDArray mapi(NDArray other, DoubleBinaryOperator operator) {
      delegate.mapi(other, operator);
      return this;
   }

   @Override
   public NDArray mapi(DoubleUnaryOperator operator) {
      delegate.mapi(operator);
      return this;
   }

   @Override
   public NDArray mapi(NDArray vector, Axis axis, DoubleBinaryOperator operator) {
      delegate.mapi(vector, axis, operator);
      return this;
   }

   @Override
   public NDArray mapiIf(DoublePredicate predicate, DoubleUnaryOperator operator) {
      delegate.mapiIf(predicate, operator);
      return this;
   }

   @Override
   public NDArray mapiSparse(NDArray vector, Axis axis, DoubleBinaryOperator operator) {
      delegate.mapiSparse(vector, axis, operator);
      return this;
   }

   @Override
   public NDArray mapiSparse(NDArray other, DoubleBinaryOperator operator) {
      delegate.mapiSparse(other, operator);
      return this;
   }

   @Override
   public NDArray mapiSparse(DoubleUnaryOperator operator) {
      delegate.mapiSparse(operator);
      return this;
   }

   @Override
   public double max() {
      return delegate.max();
   }

   @Override
   public NDArray max(Axis axis) {
      return delegate.max(axis);
   }

   @Override
   public double mean() {
      return delegate.mean();
   }

   @Override
   public NDArray mean(Axis axis) {
      return delegate.mean(axis);
   }

   @Override
   public double min() {
      return delegate.min();
   }

   @Override
   public NDArray min(Axis axis) {
      return delegate.min(axis);
   }

   @Override
   public NDArray mmul(NDArray other) {
      return delegate.mmul(other);
   }

   @Override
   public NDArray mul(double scalar) {
      return delegate.mul(scalar);
   }

   @Override
   public NDArray mul(NDArray other) {
      return delegate.mul(other);
   }

   @Override
   public NDArray mul(NDArray other, Axis axis) {
      return delegate.mul(other, axis);
   }

   @Override
   public NDArray muli(double scalar) {
      delegate.muli(scalar);
      return this;
   }

   @Override
   public NDArray muli(NDArray other) {
      delegate.muli(other);
      return this;
   }

   @Override
   public NDArray muli(NDArray other, Axis axis) {
      delegate.muli(other, axis);
      return this;
   }

   @Override
   public NDArray neg() {
      return delegate.neg();
   }

   @Override
   public NDArray negi() {
      delegate.negi();
      return this;
   }

   @Override
   public double norm1() {
      return delegate.norm1();
   }

   @Override
   public double norm2() {
      return delegate.norm2();
   }

   @Override
   public NDArray pow(double pow) {
      return delegate.pow(pow);
   }

   @Override
   public NDArray powi(double pow) {
      delegate.powi(pow);
      return this;
   }

   @Override
   public NDArray rdiv(double scalar) {
      return delegate.rdiv(scalar);
   }

   @Override
   public NDArray rdiv(NDArray other) {
      return delegate.rdiv(other);
   }

   @Override
   public NDArray rdiv(NDArray other, Axis axis) {
      return delegate.rdiv(other, axis);
   }

   @Override
   public NDArray rdivi(double scalar) {
      delegate.rdivi(scalar);
      return this;
   }

   @Override
   public NDArray rdivi(NDArray other) {
      delegate.rdivi(other);
      return this;
   }

   @Override
   public NDArray rdivi(NDArray other, Axis axis) {
      delegate.rdivi(other, axis);
      return this;
   }

   @Override
   public NDArray rmap(NDArray vector, Axis axis, DoubleBinaryOperator operator) {
      return delegate.rmap(vector, axis, operator);
   }

   @Override
   public NDArray rmap(NDArray other, DoubleBinaryOperator operator) {
      return delegate.rmap(other, operator);
   }

   @Override
   public NDArray rmapi(NDArray other, DoubleBinaryOperator operator) {
      delegate.rmapi(other, operator);
      return this;
   }

   @Override
   public NDArray rmapi(NDArray vector, Axis axis, DoubleBinaryOperator operator) {
      delegate.rmapi(vector, axis, operator);
      return this;
   }

   @Override
   public NDArray rsub(double scalar) {
      return delegate.rsub(scalar);
   }

   @Override
   public NDArray rsub(NDArray other) {
      return delegate.rsub(other);
   }

   @Override
   public NDArray rsub(NDArray other, Axis axis) {
      return delegate.rsub(other, axis);
   }

   @Override
   public NDArray rsubi(double scalar) {
      delegate.rsubi(scalar);
      return this;
   }

   @Override
   public NDArray rsubi(NDArray other) {
      delegate.rsubi(other);
      return this;
   }

   @Override
   public NDArray rsubi(NDArray other, Axis axis) {
      delegate.rsubi(other, axis);
      return this;
   }

   @Override
   public double scalarValue() {
      return delegate.scalarValue();
   }

   @Override
   public NDArray select(DoublePredicate predicate) {
      return delegate.select(predicate);
   }

   @Override
   public NDArray select(NDArray predicate) {
      return delegate.select(predicate);
   }

   @Override
   public NDArray selecti(DoublePredicate predicate) {
      delegate.selecti(predicate);
      return this;
   }

   @Override
   public NDArray selecti(NDArray predicate) {
      delegate.selecti(predicate);
      return this;
   }

   @Override
   public NDArray set(int index, double value) {
      delegate.set(index, value);
      return this;
   }

   @Override
   public NDArray set(int r, int c, double value) {
      delegate.set(r, c, value);
      return this;
   }

   @Override
   public NDArray set(Subscript subscript, double value) {
      delegate.set(subscript, value);
      return this;
   }

   @Override
   public NDArray setVector(int index, NDArray vector, Axis axis) {
      delegate.setVector(index, vector, axis);
      return this;
   }

   @Override
   public Shape shape() {
      return delegate.shape();
   }

   @Override
   public int size() {
      return delegate.size();
   }

   @Override
   public NDArray slice(int from, int to) {
      return delegate.slice(from, to);
   }

   @Override
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      return delegate.slice(iFrom, iTo, jFrom, jTo);
   }

   @Override
   public NDArray slice(Axis axis, int... indexes) {
      return delegate.slice(axis, indexes);
   }

   @Override
   public Iterator<Entry> sparseIterator() {
      return delegate.sparseIterator();
   }

   @Override
   public Iterator<Entry> sparseOrderedIterator() {
      return delegate.sparseOrderedIterator();
   }

   @Override
   public EnhancedDoubleStatistics statistics() {
      return delegate.statistics();
   }

   @Override
   public NDArray sub(double scalar) {
      return delegate.sub(scalar);
   }

   @Override
   public NDArray sub(NDArray other) {
      return delegate.sub(other);
   }

   @Override
   public NDArray sub(NDArray other, Axis axis) {
      return delegate.sub(other, axis);
   }

   @Override
   public NDArray subi(double scalar) {
      delegate.subi(scalar);
      return this;
   }

   @Override
   public NDArray subi(NDArray other) {
      delegate.subi(other);
      return this;
   }

   @Override
   public NDArray subi(NDArray other, Axis axis) {
      delegate.subi(other, axis);
      return this;
   }

   @Override
   public NDArray sum(Axis axis) {
      return delegate.sum(axis);
   }

   @Override
   public double sum() {
      return delegate.sum();
   }

   @Override
   public double sumOfSquares() {
      return delegate.sumOfSquares();
   }

   @Override
   public NDArray test(DoublePredicate predicate) {
      return delegate.test(predicate);
   }

   @Override
   public NDArray testi(DoublePredicate predicate) {
      delegate.testi(predicate);
      return this;
   }

   @Override
   public double[][] to2DArray() {
      return delegate.to2DArray();
   }

   @Override
   public double[] toArray() {
      return delegate.toArray();
   }

   @Override
   public boolean[] toBooleanArray() {
      return delegate.toBooleanArray();
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return delegate.toDoubleMatrix();
   }

   @Override
   public float[] toFloatArray() {
      return delegate.toFloatArray();
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      return delegate.toFloatMatrix();
   }

   @Override
   public int[] toIntArray() {
      return delegate.toIntArray();
   }

   @Override
   public NDArray zero() {
      delegate.zero();
      return this;
   }

}// END OF ForwardingNDArray

package com.davidbracewell.apollo.linear.sparse;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.Iterables;
import com.davidbracewell.guava.common.collect.Iterators;
import org.apache.mahout.math.function.IntDoubleProcedure;

import java.io.Serializable;
import java.util.*;
import java.util.function.Consumer;

/**
 * The type Sparse 2 d storage.
 *
 * @author David B. Bracewell
 */
public class Sparse2dStorage implements Serializable {
   private static final long serialVersionUID = 1L;
   private Node[] rows;
   private int[] cols;
   private int nRows;
   private int nColums;
   private ArrayList<Node> nodes;


   /**
    * Instantiates a new Sparse 2 d storage.
    */
   public Sparse2dStorage(int numRows, int numCols) {
      this.nodes = new ArrayList<>();
      this.nRows = numRows;
      this.nColums = numCols;
      this.cols = new int[numCols];
      Arrays.fill(this.cols, -1);
      this.rows = new Node[numRows];
   }


   /**
    * Instantiates a new Sparse 2 d storage.
    *
    * @param columns the columns
    */
   public Sparse2dStorage(Collection<NDArray> columns) {
      this(Iterables.getFirst(columns, null).length(), columns.size());
      Node[] last = new Node[Iterables.getFirst(columns, null).length()];
      int column = 0;
      for (NDArray n : columns) {
         List<Node> temp = new ArrayList<>();
         for (Iterator<NDArray.Entry> itr = n.sparseIterator(); itr.hasNext(); ) {
            NDArray.Entry e = itr.next();
            int row = e.getIndex();
            double v = e.getValue();
            if (v != 0) {
               Node node = new Node(row, column, NDArray.columnMajorIndex(row, column, nRows, nColums), v);
               if (rows[row] == null) {
                  rows[row] = node;
               }
               if (last[row] != null) {
                  last[row].nextRow = node;
               }
               last[row] = node;
               if (cols[column] == -1) {
                  cols[column] = nodes.size();
               }
               temp.add(node);
            }
         }
         nodes.addAll(temp);

         column++;
      }
   }

   private int binarySearch(int i, int j) {
      return binarySearch(NDArray.columnMajorIndex(i, j, nRows, nColums));
   }

   private int binarySearch(int index) {
      return binarySearch(index, 0, nodes.size() - 1);
   }

   private int binarySearch(int index, int low, int high) {
      while (low <= high) {
         int mid = (low + high) >>> 1;
         Node n = nodes.get(mid);
         if (n.index == index) {
            return mid;
         } else if (n.index < index) {
            low = mid + 1;
         } else {
            high = mid - 1;
         }
      }
      return -(low + 1);  // key not found
   }

   /**
    * Clear.
    */
   public void clear() {
      this.nodes.clear();
   }

   /**
    * For each.
    *
    * @param consumer the consumer
    */
   public void forEach(Consumer<NDArray.Entry> consumer) {
      nodes.forEach(consumer);
   }

   /**
    * For each pair.
    *
    * @param dbl the dbl
    */
   public void forEachPair(IntDoubleProcedure dbl) {
      for (Node n : nodes) {
         dbl.apply(n.index, n.value);
      }
   }

   /**
    * Get double.
    *
    * @param index the index
    * @return the double
    */
   public double get(int index) {
      int ii = binarySearch(index);
      if (ii >= 0) {
         return nodes.get(ii).value;
      }
      return 0d;
   }

   public NDArray.Entry getSparse(int sparseIndex) {
      return nodes.get(sparseIndex);
   }

   public Iterator<NDArray.Entry> iterator() {
      final int length = nRows * nColums;
      return new Iterator<NDArray.Entry>() {
         int index = 0;
         int sindex = 0;

         @Override
         public boolean hasNext() {
            return index < length;
         }

         @Override
         public NDArray.Entry next() {
            while (sindex < nodes.size() && index > nodes.get(sindex).index) {
               sindex++;
            }
            if (sindex < nodes.size() && index == nodes.get(sindex).index) {
               index++;
               sindex++;
               return nodes.get(sindex - 1);
            }
            NDArray.Entry e = new VirtualNode(NDArray.toRow(index, nRows, nColums),
                                              NDArray.toColumn(index, nRows, nColums),
                                              index);
            index++;
            return e;
         }
      };
   }

   public int numColumns() {
      return nColums;
   }

   public int numRows() {
      return nRows;
   }

   /**
    * Put.
    *
    * @param index the index
    * @param value the value
    */
   public void put(int index, double value) {
      putAt(binarySearch(index),
            NDArray.toRow(index, nRows, nColums),
            NDArray.toColumn(index, nRows, nColums),
            value);
   }

   /**
    * Put.
    *
    * @param i     the
    * @param j     the j
    * @param value the value
    */
   public void put(int i, int j, double value) {
      putAt(binarySearch(i, j), i, j, value);
   }

   private void putAt(int index, int row, int column, double value) {
      if (index >= 0) {
         if (value == 0) {
            Node n = nodes.remove(index);
            if (rows[n.getI()].equals(n)) {
               rows[n.getI()] = n.nextRow;
            }
         } else {
            nodes.get(index).setValue(value);
         }
      } else if (value != 0) {
         int ii = Math.abs(index + 1);
         Node newNode = new Node(row, column, NDArray.columnMajorIndex(row, column, nRows, nColums), value);
         if (cols[newNode.getJ()] == -1) {
            cols[newNode.getJ()] = ii;
         } else if (cols[newNode.getJ()] > ii) {
            cols[newNode.getJ()] = ii;
         }
         int r = newNode.getI();
         if (rows[r] == null) {
            rows[r] = newNode;
         } else if (rows[r].getJ() > newNode.getJ()) {
            newNode.nextRow = rows[r];
            rows[r] = newNode;
         } else {
            Node temp = rows[r];
            Node last = rows[r];
            while (temp != null && temp.getJ() < newNode.getJ()) {
               last = temp;
               temp = temp.nextRow;
            }
            if (temp == null) {
               last.nextRow = newNode;
            } else {
               newNode.nextRow = temp;
               last.nextRow = newNode;
            }
         }
         if (nodes.size() == 0 || ii >= nodes.size()) {
            nodes.add(newNode);
         } else {
            nodes.add(ii, newNode);
         }
      }
   }

   private void remove(Node n) {
      int index = Collections.binarySearch(nodes, n);
      if (index >= 0) {
         nodes.remove(index);
         Node row = rows[n.row];
         Node last = null;
         while (row != null && row.index != n.index) {
            last = row;
            row = row.nextRow;
         }
         if (row != null && row.index == n.index) {
            if (last == null) {
               rows[n.row] = n.nextRow;
            } else {
               last.nextRow = n.nextRow;
            }
         }
         if (cols[n.column] == index) {
            Node nc = null;
            if (index + 1 < nodes.size() && nodes.get(index + 1).column == n.column) {
               nc = nodes.get(index + 1);
            }
            if (nc != null) {
               cols[n.column] = index + 1;
            } else {
               cols[n.column] = -1;
            }
         }
      }
   }

   public void reshape(int numRows, int numColumns) {
      Preconditions.checkArgument(numRows * numColumns == nRows * nColums, "Length cannot change");
      this.nRows = numRows;
      this.nColums = numColumns;
      this.rows = new Node[numRows];
      this.cols = new int[numColumns];
      Arrays.fill(this.cols, -1);

      Node[] last = new Node[numRows];
      for (int i = 0; i < nodes.size(); i++) {
         Node n = nodes.get(i);

         int r = n.row;
         int c = n.column;

         if (cols[c] == -1) {
            cols[c] = i;
         }

         if (rows[r] == null) {
            rows[r] = n;
         }
         if (last[r] != null) {
            last[r].nextRow = n;
         }

         last[r] = n;
      }

   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return nodes.size();
   }

   /**
    * Sparse column iterator.
    *
    * @param col the col
    * @return the iterator
    */
   public Iterator<NDArray.Entry> sparseColumn(final int col) {
      return new SparseColumnIterator(col);
   }

   /**
    * Iterator iterator.
    *
    * @return the iterator
    */
   public Iterator<NDArray.Entry> sparseIterator() {
      return Iterators.unmodifiableIterator(Cast.cast(nodes.iterator()));
   }

   /**
    * Sparse row iterator.
    *
    * @param row the row
    * @return the iterator
    */
   public Iterator<NDArray.Entry> sparseRow(final int row) {
      return new Iterator<NDArray.Entry>() {
         Node node = row < rows.length ? rows[row] : null;

         @Override
         public boolean hasNext() {
            return node != null;
         }

         @Override
         public NDArray.Entry next() {
            Node temp = node;
            node = node.nextRow;
            return temp;
         }
      };
   }

   /**
    * Sum double.
    *
    * @return the double
    */
   public double sum() {
      return nodes.stream().mapToDouble(Node::getValue).sum();
   }

   /**
    * Trim to size.
    */
   public void trimToSize() {
      nodes.trimToSize();
   }

   /**
    * Values double [ ].
    *
    * @return the double [ ]
    */
   public double[] values() {
      return nodes.stream().mapToDouble(n -> n.value).toArray();
   }

   private class VirtualNode implements NDArray.Entry {
      final int i;
      final int j;
      final int index;
      Double value = null;

      private VirtualNode(int i, int j, int index) {
         this.i = i;
         this.j = j;
         this.index = index;
      }

      @Override
      public int getI() {
         return i;
      }

      @Override
      public int getIndex() {
         return index;
      }

      @Override
      public int getJ() {
         return j;
      }

      @Override
      public double getValue() {
         return value == null ? 0d : value;
      }

      @Override
      public void setValue(double value) {
         this.value = value;
         put(index, value);
      }
   }

   private class Node implements NDArray.Entry, Serializable, Comparable<Node> {
      private static final long serialVersionUID = 1L;
      private final int index;
      private final int row;
      private final int column;
      private double value;
      private Node nextRow;
      private boolean zeroed = false;

      /**
       * Instantiates a new Node.
       *
       * @param index the index
       * @param value the value
       */
      public Node(int i, int j, int index, double value) {
         this.row = i;
         this.column = j;
         this.index = index;
         this.value = value;
      }

      @Override
      public int compareTo(Node o) {
         return Integer.compare(index, o.index);
      }

      @Override
      public int getI() {
         return row;
      }

      @Override
      public int getIndex() {
         return index;
      }

      @Override
      public int getJ() {
         return column;
      }

      @Override
      public double getValue() {
         return value;
      }

      @Override
      public void setValue(double value) {
         if (value != this.value) {
            this.value = value;
//            if (value == 0) {
//               zeroed = true;
//               remove(this);
//               //remove
//            } else if (zeroed){
//               zeroed = false;
//               //insert
//            }
         }
      }

      @Override
      public String toString() {
         return "(" + row + ", " + column + ", " + value + ")";
      }
   }

   private class SparseColumnIterator implements Iterator<NDArray.Entry> {
      /**
       * The Column.
       */
      final int column;
      /**
       * The Row.
       */
      int row = 0;
      /**
       * The Index.
       */
      int index = -1;
      /**
       * The Last index.
       */
      int lastIndex = -1;

      /**
       * Instantiates a new Sparse column iterator.
       *
       * @param column the column
       */
      public SparseColumnIterator(int column) {
         this.column = column;
         this.index = cols[column];
         if (this.index >= 0) {
            this.row = nodes.get(this.index).getI();
         } else {
            this.row = nRows;
         }
      }

      private boolean advance() {
         if (index >= 0) {
            return true;
         }
         if (nodes.isEmpty() || lastIndex + 1 >= nodes.size() || row >= nRows) {
            return false;
         }
         index = lastIndex + 1;
         Node n = nodes.get(index);
         if (n.getJ() != column) {
            row = nRows;
            index = -1;
            return false;
         }
         row = n.getI();
         return true;
      }

      @Override
      public boolean hasNext() {
         return advance();
      }

      @Override
      public NDArray.Entry next() {
         advance();
         lastIndex = index;
         index = -1;
         return nodes.get(lastIndex);
      }
   }

}// END OF Sparse2dArray

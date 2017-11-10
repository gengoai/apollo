package com.davidbracewell.apollo.linear;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.collect.Iterators;
import lombok.Getter;
import org.apache.mahout.math.function.IntDoubleProcedure;

import java.io.Serializable;
import java.util.*;
import java.util.function.Consumer;

/**
 * The type Sparse 2 d storage.
 *
 * @author David B. Bracewell
 */
public class Sparse2dStorage {
   @Getter
   private final Shape shape;
   private final Node[] rows;
   private final int[] cols;
   private ArrayList<Node> nodes;

   /**
    * Instantiates a new Sparse 2 d storage.
    *
    * @param shape the shape
    */
   public Sparse2dStorage(Shape shape) {
      this.nodes = new ArrayList<>();
      this.shape = shape;
      this.cols = new int[shape.j];
      Arrays.fill(this.cols, -1);
      this.rows = new Node[shape.i];
   }

   /**
    * Instantiates a new Sparse 2 d storage.
    *
    * @param columns the columns
    */
   public Sparse2dStorage(NDArray... columns) {
      this(Shape.shape(columns[0].length(), columns.length));
      Node[] last = new Node[columns[0].length()];
      for (int column = 0; column < columns.length; column++) {
         List<Node> temp = new ArrayList<>();
         NDArray n = columns[column];
         for (Iterator<NDArray.Entry> itr = n.sparseIterator(); itr.hasNext(); ) {
            NDArray.Entry e = itr.next();
            int row = e.getIndex();
            double v = e.getValue();
            if (v != 0) {
               Node node = new Node(row, column, shape.colMajorIndex(row, column), v);
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
      }
   }


   private int binarySearch(int i, int j) {
      return binarySearch(shape.colMajorIndex(i, j));
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

   /**
    * Get double.
    *
    * @param i the
    * @param j the j
    * @return the double
    */
   public double get(int i, int j) {
      Node n = rows[i];
      while (n != null) {
         if (n.getJ() == j) {
            return n.value;
         } else if (n.getJ() < j) {
            return 0d;
         }
         n = n.nextRow;
      }
      return 0d;
   }

   public NDArray.Entry getSparse(int sparseIndex) {
      return nodes.get(sparseIndex);
   }

   /**
    * Iterator iterator.
    *
    * @return the iterator
    */
   public Iterator<NDArray.Entry> iterator() {
      return Iterators.unmodifiableIterator(Cast.cast(nodes.iterator()));
   }

   /**
    * Put.
    *
    * @param index the index
    * @param value the value
    */
   public void put(int index, double value) {
      putAt(binarySearch(index), shape.fromColMajorIndex(index), value);
   }

   /**
    * Put.
    *
    * @param i     the
    * @param j     the j
    * @param value the value
    */
   public void put(int i, int j, double value) {
      putAt(binarySearch(i, j), Subscript.from(i, j), value);
   }


   private void putAt(int index, Subscript si, double value) {
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
         Node newNode = new Node(si.i, si.j, shape.colMajorIndex(si), value);
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

   private class Node implements NDArray.Entry, Serializable, Comparable<Node> {
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
            this.row = shape.i;
         }
      }

      private boolean advance() {
         if (index >= 0) {
            return true;
         }
         if (nodes.isEmpty() || lastIndex + 1 >= nodes.size() || row >= shape.i) {
            return false;
         }
         index = lastIndex + 1;
         Node n = nodes.get(index);
         if (n.getJ() != column) {
            row = shape.i;
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

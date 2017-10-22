package com.davidbracewell.apollo.linear;

import lombok.Getter;
import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.list.IntArrayList;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class Sparse2dArray {
   @Getter
   private final Shape shape;
   private final Node[] rows;
   private final int[] cols;
   private ArrayList<Node> nodes;

   public Sparse2dArray(Shape shape) {
      this.nodes = new ArrayList<>();
      this.shape = shape;
      this.cols = new int[shape.j];
      Arrays.fill(this.cols, -1);
      this.rows = new Node[shape.i];
   }

   private int binarySearch(Subscript si) {
      return binarySearch(shape.colMajorIndex(si));
   }

   private int binarySearch(int i, int j) {
      return binarySearch(shape.colMajorIndex(i, j));
   }

   private int binarySearch(int index) {
      return binarySearch(index, 0, nodes.size() - 1);
   }

   private int binarySearch(int i, int j, int low, int high) {
      return binarySearch(shape.colMajorIndex(i, j), low, high);
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


   public void clear() {
      this.nodes.clear();
   }

   public void forEach(Consumer<NDArray.Entry> consumer) {
      nodes.forEach(consumer);
   }

   public void forEachPair(IntDoubleProcedure dbl) {
      for (Node n : nodes) {
         dbl.apply(n.index, n.value);
      }
   }

   public double get(int index) {
      int ii = binarySearch(index);
      if (ii >= 0) {
         return nodes.get(ii).value;
      }
      return 0d;
   }

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

   private int[] indexes() {
      return nodes.stream().mapToInt(n -> n.index)
                  .toArray();
   }

   public IntArrayList keys() {
      return new IntArrayList(indexes());
   }

   public double max() {
      return nodes.stream()
                  .mapToDouble(Node::getValue)
                  .max().orElse(Double.NEGATIVE_INFINITY);
   }

   public double min() {
      return nodes.stream()
                  .mapToDouble(Node::getValue)
                  .min().orElse(Double.POSITIVE_INFINITY);
   }

   public void put(int index, double value) {
      putAt(binarySearch(index), shape.fromColMajorIndex(index), value);
   }

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
         Node newNode = new Node(si, shape.colMajorIndex(si), value);
         if (cols[newNode.getJ()] == -1) {
            cols[newNode.getJ()] = newNode.index;
         } else if (cols[newNode.getJ()] > newNode.index) {
            cols[newNode.getJ()] = newNode.index;
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

   public int size() {
      return nodes.size();
   }

   public Iterator<NDArray.Entry> sparseColumn(final int col) {
      return new SparseColumnIterator(col);
   }

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

   public List<Subscript> subscripts() {
      return nodes.stream().map(n -> n.si)
                  .collect(Collectors.toList());
   }

   public double sum() {
      return nodes.stream()
                  .mapToDouble(Node::getValue)
                  .sum();
   }

   public void trimToSize() {
      nodes.trimToSize();
   }

   public double[] values() {
      return nodes.stream().mapToDouble(n -> n.value)
                  .toArray();
   }

   private static class Node implements NDArray.Entry, Serializable {
      private final Subscript si;
      private final int index;
      private double value;
      private Node nextRow;

      public Node(Subscript si, int index, double value) {
         this.si = si;
         this.index = index;
         this.value = value;
      }

      @Override
      public int getI() {
         return si.i;
      }

      @Override
      public int getIndex() {
         return index;
      }

      @Override
      public int getJ() {
         return si.j;
      }

      @Override
      public double getValue() {
         return value;
      }

      @Override
      public void setValue(double value) {
         this.value = value;
      }

      @Override
      public String toString() {
         return "(" + si.i + ", " + si.j + ", " + value + ")";
      }
   }

   private class SparseColumnIterator implements Iterator<NDArray.Entry> {
      final int column;
      int row = 0;
      int index = -1;
      int lastIndex = -1;

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
            row = shape.j;
            index = -1;
            return false;
         }
         row = n.getI();
         return true;
//         if (row == 0) { //initial step
//            index = binarySearch(shape.colMajorIndex(0, column));
//            if (index >= 0) {
//               return true;
//            }
//            int aindex = Math.abs(index);
//            if (aindex >= nodes.size()) {
//               index = nodes.size();
//               return false;
//            }
//            if (nodes.get(aindex).getJ() == column) {
//               row = nodes.get(aindex).getI();
//               index = nodes.get(aindex).index;
//               return true;
//            }
//            row = shape.i;
//            return false;
//         }
//         index = lastIndex + 1;
//         return index < nodes.size() && nodes.get(index).getJ() == column;
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

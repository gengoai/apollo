package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class Cluster implements Serializable, Iterable<NDArray> {
   private static final long serialVersionUID = 1L;
   private final List<NDArray> points = new ArrayList<>();
   private NDArray centroid;
   private int id;
   private Cluster left;
   private Cluster parent;
   private Cluster right;
   private double score;

   public void addPoint(NDArray point){
      this.points.add(point);
   }

   /**
    * Clears the cluster, removing all points
    */
   public void clear() {
      points.clear();
   }

   public NDArray getCentroid() {
      return centroid;
   }

   public void setCentroid(NDArray centroid) {
      this.centroid = centroid;
   }

   public int getId() {
      return id;
   }

   public void setId(int id) {
      this.id = id;
   }

   public Cluster getLeft() {
      return left;
   }

   public void setLeft(Cluster left) {
      this.left = left;
   }

   public Cluster getParent() {
      return parent;
   }

   public void setParent(Cluster parent) {
      this.parent = parent;
   }

   public List<NDArray> getPoints() {
      return points;
   }

   public Cluster getRight() {
      return right;
   }

   public void setRight(Cluster right) {
      this.right = right;
   }

   public double getScore() {
      return score;
   }

   public void setScore(double score) {
      this.score = score;
   }

   /**
    * Gets the score of the given vector respective to the cluster
    *
    * @param point The point whose score we want
    */
   public double getScore(NDArray point) {
      return points.contains(point) ? 1.0 : 0.0;
   }

   @Override
   public Iterator<NDArray> iterator() {
      return points.iterator();
   }

   /**
    * The number of points in the cluster
    *
    * @return the number of points in the cluster
    */
   public int size() {
      return points.size();
   }

   @Override
   public String toString() {
      return "Cluster(id=" + id + ", size=" + points.size() + ")";
   }


}//END OF Cluster

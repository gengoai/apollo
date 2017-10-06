/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.linear.NDArray;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * <p>Represents a cluster which minimally is a collection of vectors. May also include a centroid and in the case of
 * hierarchical clusters a parent, left, and right child.</p>
 *
 * @author David B. Bracewell
 */
public class Cluster implements Serializable, Iterable<NDArray> {

   private static final long serialVersionUID = 1L;
   private final List<NDArray> points = new ArrayList<>();
   @Getter
   @Setter
   private NDArray centroid;
   @Getter
   @Setter
   private Cluster parent;
   @Getter
   @Setter
   private Cluster left;
   @Getter
   @Setter
   private Cluster right;
   @Getter
   @Setter
   private double score;
   @Getter
   @Setter
   private int id;

   /**
    * Adds a point to the cluster
    *
    * @param point the point
    */
   public void addPoint(final NDArray point) {
      if (point != null) {
         points.add(point);
      }
   }

   /**
    * Clears the cluster, removing all points
    */
   public void clear() {
      points.clear();
   }

   /**
    * Gets the points in the cluster as a list.
    *
    * @return the points as a list
    */
   public List<NDArray> getPoints() {
      return points;
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

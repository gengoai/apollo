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

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.FeatureVector;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

/**
 * The type Cluster.
 *
 * @author David B. Bracewell
 */
public class Cluster implements Serializable, Iterable<FeatureVector> {

  private static final long serialVersionUID = 1L;
  private final List<FeatureVector> points = new CopyOnWriteArrayList<>();
  private Vector centroid;
  private Cluster parent;
  private Cluster left;
  private Cluster right;
  private double score;

  /**
   * Add point.
   *
   * @param point the point
   */
  public void addPoint(final FeatureVector point) {
    if (point != null) {
      points.add(point);
    }
  }

  /**
   * Gets points.
   *
   * @return the points
   */
  public List<FeatureVector> getPoints() {
    return points;
  }

  @Override
  public Iterator<FeatureVector> iterator() {
    return points.iterator();
  }

  /**
   * Clear void.
   */
  public void clear() {
    points.clear();
  }

  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return points.size();
  }

  /**
   * Gets left.
   *
   * @return the left
   */
  public Cluster getLeft() {
    return left;
  }

  /**
   * Sets left.
   *
   * @param left the left
   */
  public void setLeft(Cluster left) {
    this.left = left;
  }

  /**
   * Gets parent.
   *
   * @return the parent
   */
  public Cluster getParent() {
    return parent;
  }

  /**
   * Sets parent.
   *
   * @param parent the parent
   */
  public void setParent(Cluster parent) {
    this.parent = parent;
  }

  /**
   * Gets right.
   *
   * @return the right
   */
  public Cluster getRight() {
    return right;
  }

  /**
   * Sets right.
   *
   * @param right the right
   */
  public void setRight(Cluster right) {
    this.right = right;
  }

  /**
   * Gets score.
   *
   * @return the score
   */
  public double getScore() {
    return score;
  }

  /**
   * Sets score.
   *
   * @param score the score
   */
  public void setScore(double score) {
    this.score = score;
  }

  public Vector getCentroid() {
    return centroid;
  }

  public void setCentroid(Vector centroid) {
    this.centroid = centroid;
  }

  @Override
  public String toString() {
    return points.stream().map(FeatureVector::getDecodedLabel).collect(Collectors.toList()).toString();
  }

}//END OF Cluster

/*
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

package com.gengoai.apollo.ml.model;

import com.gengoai.apollo.ml.DataSet;
import com.gengoai.apollo.ml.Datum;
import com.gengoai.apollo.ml.encoder.Encoder;
import com.gengoai.apollo.ml.transform.Transformer;
import com.gengoai.collection.Sets;
import com.gengoai.io.Compression;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.json.Json;
import com.gengoai.reflection.Reflect;
import com.gengoai.reflection.ReflectionException;
import lombok.NonNull;
import org.tensorflow.SavedModelBundle;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * <p>Abstract base class wrapping models trained in TensorFlow (Python) and exported using the TensorFlow Serving
 * Model format. The organization of a TensorFlow model directory is as follows:</p>
 * <ul>
 *    <li>tfmodel - Folder containing the exported Python model (saved_model.pb, variables)</li>
 *    <li>*.encoder.json.gz - One gzipped json dump per Observation</li>
 *    <li>__class__ - Plain text file with the fully qualified class name of the model implementation.</li>
 * </ul>
 * <p>To train a TensorFlow model you must:</p>
 * <ol>
 *    <li>Create a subclass of TensorFlowModel - Define the <code>createTransformer</code> and <code>process</code> methods.</li>
 *    <li>Generate a json dump of your training data by loading the DataSet and call estimate on your subclass. Note the location of where the dataset was saved.</li>
 *    <li>Train your Python model off the dumped dataset</li>
 *    <li>Copy exported serving model to a tfmodel directory where you saved your trained Java Model</li>
 *    <li>Profit!</li>
 * </ol>
 */
public abstract class TensorFlowModel implements Model {
   private static final long serialVersionUID = 1L;
   protected final Map<String, Encoder> encoders = new HashMap<>();
   private final Set<String> inputs;
   private final Set<String> outputs;
   private final FitParameters<?> fitParameters = new FitParameters<>();
   protected Resource modelFile;
   protected volatile transient Transformer transformer;
   private volatile transient SavedModelBundle model;

   /**
    * Reads the model from the given resource
    *
    * @param resource the resource
    * @return the model
    * @throws IOException Something went wrong reading the model
    */
   public static Model load(@NonNull Resource resource) throws IOException {
      Class<?> modelClass = Reflect.getClassForNameQuietly(resource.getChild("__class__").readToString().strip());
      try {
         TensorFlowModel m = Reflect.onClass(modelClass).allowPrivilegedAccess().create().get();
         for(Resource child : resource.getChildren("*.encoder.json.gz")) {
            String name = child.baseName().replace(".encoder.json.gz", "").strip();
            m.encoders.put(name, Json.parse(child, Encoder.class));
         }
         m.modelFile = resource;
         return m;
      } catch(ReflectionException e) {
         throw new IOException(e);
      }
   }

   /**
    * Instantiates a new TensorFlowModel.
    *
    * @param outputs  the model outputs
    * @param encoders the encoders
    */
   protected TensorFlowModel(@NonNull Set<String> outputs,
                             @NonNull Map<String, Encoder> encoders) {
      this.encoders.putAll(encoders);
      this.inputs = new HashSet<>(Sets.difference(encoders.keySet(), outputs));
      this.outputs = outputs;
   }

   /**
    * Creates the required transformer for preparing inputs / outputs to pass to TensorFlow.
    *
    * @return the transformer
    */
   protected abstract Transformer createTransformer();

   @Override
   public void estimate(@NonNull DataSet dataset) {
      dataset = createTransformer().fitAndTransform(dataset);
      dataset.getMetadata()
             .forEach((k, v) -> encoders.put(k, v.getEncoder()));
      Resource tmp = Resources.temporaryFile();
      try {
         Json.dumpPretty(dataset, tmp);
      } catch(IOException e) {
         throw new RuntimeException(e);
      }
      System.out.println("DataSet saved to: " + tmp.descriptor());
   }

   @Override
   public FitParameters<?> getFitParameters() {
      return fitParameters;
   }

   @Override
   public Set<String> getInputs() {
      return inputs;
   }

   @Override
   public Set<String> getOutputs() {
      return outputs;
   }

   protected final SavedModelBundle getTensorFlowModel() {
      if(model == null) {
         synchronized(this) {
            if(model == null) {
               model = SavedModelBundle.load(modelFile.getChild("tfmodel")
                                                      .asFile()
                                                      .orElseThrow()
                                                      .getAbsolutePath(),
                                             "serve");
               transformer = createTransformer();
            }
         }
      }
      return model;
   }

   /**
    * Processes the given datum with the model.
    *
    * @param datum the datum
    * @param model the model
    */
   protected abstract void process(@NonNull Datum datum, @NonNull SavedModelBundle model);

   @Override
   public void save(@NonNull Resource resource) throws IOException {
      for(String name : Sets.union(getInputs(), getOutputs())) {
         Encoder encoder = encoders.get(name);
         if(!encoder.isFixed()) {
            Json.dumpPretty(encoder, resource.getChild(name + ".encoder.json.gz").setCompression(Compression.GZIP));
         }
      }
   }

   @Override
   public final Datum transform(@NonNull Datum datum) {
      getTensorFlowModel();
      datum = transformer.transform(datum);
      process(datum, model);
      return datum;
   }

}//END OF TensorFlowModel

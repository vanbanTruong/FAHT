/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    GaussianConditionalSufficientStats.java
 *    Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.ht;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import weka.classifiers.trees.ht.GaussianConditionalSufficientStats.GaussianEstimator;
import weka.core.Utils;
import weka.estimators.UnivariateNormalEstimator;

/**
 * Maintains sufficient stats for a Gaussian distribution for a numeric
 * attribute
 * 
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public class GaussianConditionalSufficientStats extends ConditionalSufficientStats implements Serializable {

	/**
	 * For serialization
	 */
	private static final long serialVersionUID = -1527915607201784762L;

	/**
	 * Inner class that implements a Gaussian estimator
	 * 
	 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
	 */
	protected class GaussianEstimator extends UnivariateNormalEstimator implements Serializable {

		/**
		 * For serialization
		 */
		private static final long serialVersionUID = 4756032800685001315L;

		public double getSumOfWeights() {
			return m_SumOfWeights;
		}

		public double probabilityDensity(double value) {
			updateMeanAndVariance();

			if (m_SumOfWeights > 0) {
				double stdDev = Math.sqrt(m_Variance);
				if (stdDev > 0) {
					double diff = value - m_Mean;
					return (1.0 / (CONST * stdDev)) * Math.exp(-(diff * diff / (2.0 * m_Variance)));
				}
				return value == m_Mean ? 1.0 : 0.0;
			}

			return 0.0;
		}

		public double[] weightLessThanEqualAndGreaterThan(double value) {
			double stdDev = Math.sqrt(m_Variance);
			double equalW = probabilityDensity(value) * m_SumOfWeights;

			double lessW = (stdDev > 0)
					? weka.core.Statistics.normalProbability((value - m_Mean) / stdDev) * m_SumOfWeights - equalW
					: (value < m_Mean) ? m_SumOfWeights - equalW : 0.0;
			double greaterW = m_SumOfWeights - equalW - lessW;

			return new double[] { lessW, equalW, greaterW };
		}
	}

	protected Map<String, Double> m_minValObservedPerClass = new HashMap<String, Double>();
	protected Map<String, Double> m_maxValObservedPerClass = new HashMap<String, Double>();

	/** min and max values for each discrimination related normal distribution */
	protected Map<String, Double> minValObservedPerGauss = new HashMap<String, Double>() {
		{
			put(deprivedTotalDist, Double.POSITIVE_INFINITY);
			put(undeprivedTotalDist, Double.POSITIVE_INFINITY);
			put(deprivedGrantedDist, Double.POSITIVE_INFINITY);
			put(undeprivedGrantedDist, Double.POSITIVE_INFINITY);
		}
	};
	protected Map<String, Double> maxValObservedPerGauss = new HashMap<String, Double>() {
		{
			put(deprivedTotalDist, Double.NEGATIVE_INFINITY);
			put(undeprivedTotalDist, Double.NEGATIVE_INFINITY);
			put(deprivedGrantedDist, Double.NEGATIVE_INFINITY);
			put(undeprivedGrantedDist, Double.NEGATIVE_INFINITY);
		}
	};

	protected int m_numBins = 10;

	public void setNumBins(int b) {
		m_numBins = b;
	}

	public int getNumBins() {
		return m_numBins;
	}

	@Override
	public void update(double attVal, String classVal, double weight, int senAttVal) {
		if (!Utils.isMissingValue(attVal)) {

			// update normal distributions for different class values, i.e., m_classLookup
			GaussianEstimator norm = (GaussianEstimator) m_classLookup.get(classVal);
			if (norm == null) {
				norm = new GaussianEstimator();
				m_classLookup.put(classVal, norm);
				m_minValObservedPerClass.put(classVal, attVal);
				m_maxValObservedPerClass.put(classVal, attVal);
			} else {
				if (attVal < m_minValObservedPerClass.get(classVal)) {
					m_minValObservedPerClass.put(classVal, attVal);
				}

				if (attVal > m_maxValObservedPerClass.get(classVal)) {
					m_maxValObservedPerClass.put(classVal, attVal);
				}
			}
			norm.addValue(attVal, weight);

			// update numGaussDistLookup
			if (senAttVal == undeprivedIndex) {
				// update undeprivedTotalDist
				updateNumGaussDistLookup(attVal, weight, undeprivedTotalDist);
				if (classVal.equals(grantedString)) {
					// update undeprivedGrantedDist
					updateNumGaussDistLookup(attVal, weight, undeprivedGrantedDist);
				}
			}

			if (senAttVal == deprivedIndex) {
				// update deprivedTotalDist
				updateNumGaussDistLookup(attVal, weight, deprivedTotalDist);
				if (classVal.equals(grantedString)) {
					// update deprivedGrantedDist
					updateNumGaussDistLookup(attVal, weight, deprivedGrantedDist);
				}
			}

		}
	}

	/**
	 * update one of the four entries of numGaussDistLookup
	 * @param attVal
	 * @param weight
	 * @param whichToUpdate: which entry of numGaussDistLookup to be updated
	 */
	public void updateNumGaussDistLookup(double attVal, double weight, String whichToUpdate) {
		GaussianEstimator discNorm = (GaussianEstimator) numGaussDistLookup.get(whichToUpdate);
		if (discNorm == null) {
			discNorm = new GaussianEstimator();
			numGaussDistLookup.put(whichToUpdate, discNorm);
			minValObservedPerGauss.put(whichToUpdate, attVal);
			maxValObservedPerGauss.put(whichToUpdate, attVal);
		} else {
			if (attVal < minValObservedPerGauss.get(whichToUpdate)) {
				minValObservedPerGauss.put(whichToUpdate, attVal);
			}
			if (attVal > maxValObservedPerGauss.get(whichToUpdate)) {
				maxValObservedPerGauss.put(whichToUpdate, attVal);
			}
		}
		discNorm.addValue(attVal, weight);
	}

	@Override
	public double probabilityOfAttValConditionedOnClass(double attVal, String classVal) {
		GaussianEstimator norm = (GaussianEstimator) m_classLookup.get(classVal);
		if (norm == null) {
			return 0;
		}

		// return Utils.lo
		return norm.probabilityDensity(attVal);
	}

	protected TreeSet<Double> getSplitPointCandidates() {
		TreeSet<Double> splits = new TreeSet<Double>();
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;

		for (String classVal : m_classLookup.keySet()) {
			if (m_minValObservedPerClass.containsKey(classVal)) {
				if (m_minValObservedPerClass.get(classVal) < min) {
					min = m_minValObservedPerClass.get(classVal);
				}

				if (m_maxValObservedPerClass.get(classVal) > max) {
					max = m_maxValObservedPerClass.get(classVal);
				}
			}
		}

		if (min < Double.POSITIVE_INFINITY) {
			double bin = max - min;
			bin /= (m_numBins + 1);
			for (int i = 0; i < m_numBins; i++) {
				double split = min + (bin * (i + 1));

				if (split > min && split < max) {
					splits.add(split);
				}
			}
		}
		return splits;
	}

	protected List<Map<String, WeightMass>> classDistsAfterSplit(double splitVal) {
		Map<String, WeightMass> lhsDist = new HashMap<String, WeightMass>();
		Map<String, WeightMass> rhsDist = new HashMap<String, WeightMass>();

		for (Map.Entry<String, Object> e : m_classLookup.entrySet()) {
			String classVal = e.getKey();
			GaussianEstimator attEst = (GaussianEstimator) e.getValue();

			if (attEst != null) {
				if (splitVal < m_minValObservedPerClass.get(classVal)) {
					WeightMass mass = rhsDist.get(classVal);
					if (mass == null) {
						mass = new WeightMass();
						rhsDist.put(classVal, mass);
					}
					mass.m_weight += attEst.getSumOfWeights();
				} else if (splitVal > m_maxValObservedPerClass.get(classVal)) {
					WeightMass mass = lhsDist.get(classVal);
					if (mass == null) {
						mass = new WeightMass();
						lhsDist.put(classVal, mass);
					}
					mass.m_weight += attEst.getSumOfWeights();
				} else {
					double[] weights = attEst.weightLessThanEqualAndGreaterThan(splitVal);
					WeightMass mass = lhsDist.get(classVal);
					if (mass == null) {
						mass = new WeightMass();
						lhsDist.put(classVal, mass);
					}
					mass.m_weight += weights[0] + weights[1]; // <=

					mass = rhsDist.get(classVal);
					if (mass == null) {
						mass = new WeightMass();
						rhsDist.put(classVal, mass);
					}
					mass.m_weight += weights[2]; // >
				}
			}
		}

		List<Map<String, WeightMass>> dists = new ArrayList<Map<String, WeightMass>>();
		dists.add(lhsDist);
		dists.add(rhsDist);

		return dists;
	}
	
	/**
	 *  calculate discrimination based on splitVal
	 */
	public double splitValDiscrimination(double splitVal) {
		// left part of the splitVal
		Map<String, WeightMass> leftDist = new HashMap<String, WeightMass>() {
			{
				put(deprivedTotalDist, null);
				put(undeprivedTotalDist, null);
				put(deprivedGrantedDist, null);
				put(undeprivedGrantedDist, null);
			}
		};
		
		// right part of the splitVal
		Map<String, WeightMass> rightDist = new HashMap<String, WeightMass>() {
			{
				put(deprivedTotalDist, null);
				put(undeprivedTotalDist, null);
				put(deprivedGrantedDist, null);
				put(undeprivedGrantedDist, null);
			}
		};

		// obtain related discrimination counts for left and right splits
		for (Map.Entry<String, Object> e : numGaussDistLookup.entrySet()) {
			String discValue = e.getKey(); // one of the four values used to calculate discrimination
			GaussianEstimator discEst = (GaussianEstimator) e.getValue();

			if (discEst != null) {
				if (splitVal < minValObservedPerGauss.get(discValue)) {
					WeightMass mass = rightDist.get(discValue);
					if (mass == null) {
						mass = new WeightMass();
						rightDist.put(discValue, mass);
					}
					mass.m_weight += discEst.getSumOfWeights();
				} else if (splitVal > maxValObservedPerGauss.get(discValue)) {
					WeightMass mass = leftDist.get(discValue);
					if (mass == null) {
						mass = new WeightMass();
						leftDist.put(discValue, mass);
					}
					mass.m_weight += discEst.getSumOfWeights();
				} else {
					double[] weights = discEst.weightLessThanEqualAndGreaterThan(splitVal);
					WeightMass mass = leftDist.get(discValue);
					if (mass == null) {
						mass = new WeightMass();
						leftDist.put(discValue, mass);
					}
					mass.m_weight += weights[0] + weights[1]; // <=

					mass = rightDist.get(discValue);
					if (mass == null) {
						mass = new WeightMass();
						rightDist.put(discValue, mass);
					}
					mass.m_weight += weights[2]; // >
				}
			} else {
				/*
				 * handled when getting the weights of left and right part by checking if it's null
				 */
			}

		}

		// left part of the split
		double lUndeprivedTotalCount= 0, lUndeprivedGrantedCount= 0, lDeprivedGrantedCount= 0, lDeprivedTotalCount= 0;
		double lUndeprivedRate = 0;
		double lDeprivedRate = 0;
		double lDiscrimination = 0;
		
		if (leftDist.get(undeprivedTotalDist)!= null) {
			lUndeprivedTotalCount = leftDist.get(undeprivedTotalDist).m_weight;
		}
		if (leftDist.get(undeprivedGrantedDist)!= null) {
			lUndeprivedGrantedCount = leftDist.get(undeprivedGrantedDist).m_weight;
		}
		if (leftDist.get(deprivedGrantedDist)!= null) {
			lDeprivedGrantedCount = leftDist.get(deprivedGrantedDist).m_weight;
		}
		if (leftDist.get(deprivedTotalDist)!= null) {
			lDeprivedTotalCount = leftDist.get(deprivedTotalDist).m_weight;
		}
		
		if (lUndeprivedTotalCount != 0) {
			lUndeprivedRate = lUndeprivedGrantedCount / lUndeprivedTotalCount;
		}

		if (lDeprivedTotalCount != 0) {
			lDeprivedRate = lDeprivedGrantedCount / lDeprivedTotalCount;
		}

		lDiscrimination = Math.abs(lUndeprivedRate - lDeprivedRate);

		// right part of the split
		double rUndeprivedTotalCount= 0, rUndeprivedGrantedCount= 0, rDeprivedGrantedCount= 0, rDeprivedTotalCount= 0;
		double rUndeprivedRate = 0;
		double rDeprivedRate = 0;
		double rDiscrimination = 0;
		
		if (rightDist.get(undeprivedTotalDist)!= null) {
			rUndeprivedTotalCount = rightDist.get(undeprivedTotalDist).m_weight;
		}
		if (rightDist.get(undeprivedGrantedDist)!= null) {
			rUndeprivedGrantedCount = rightDist.get(undeprivedGrantedDist).m_weight;
		}
		if (rightDist.get(deprivedGrantedDist)!= null) {
			rDeprivedGrantedCount = rightDist.get(deprivedGrantedDist).m_weight;
		}
		if (rightDist.get(deprivedTotalDist)!= null) {
			rDeprivedTotalCount = rightDist.get(deprivedTotalDist).m_weight;
		}

		if (rUndeprivedTotalCount != 0) {
			rUndeprivedRate = rUndeprivedGrantedCount / rUndeprivedTotalCount;
		}

		if (rDeprivedTotalCount != 0) {
			rDeprivedRate = rDeprivedGrantedCount / rDeprivedTotalCount;
		}

		rDiscrimination = Math.abs(rUndeprivedRate - rDeprivedRate);

		return Math.abs(lDiscrimination + rDiscrimination);
	}
	
	

	@Override
	public SplitCandidate bestSplit(SplitMetric splitMetric, Map<String, WeightMass> preSplitDist, String attName,
			ActiveHNode node) {

		SplitCandidate best = null;

		TreeSet<Double> candidates = getSplitPointCandidates();
		/**
		 * 1. merit is also being used when deciding the difference between first and
		 * second best; 2. Pre and post discrimination are being calculated in
		 * InfoGainSplitMetricpreDiscrimination 3. Pre discriminationthe is same for
		 * numeric attribute and nominal attribute splitting
		 */

		for (Double s : candidates) {
			List<Map<String, WeightMass>> postSplitDists = classDistsAfterSplit(s);

			double postDiscMerit = 0;
			postDiscMerit= splitValDiscrimination(s);

			double splitMerit = splitMetric.evaluateSplit(preSplitDist, postSplitDists, attName, postDiscMerit, node);

			if (best == null || splitMerit > best.m_splitMerit) {
				Split split = new UnivariateNumericBinarySplit(attName, s);
				best = new SplitCandidate(split, postSplitDists, splitMerit);
			}
		}

		return best;
	}
}

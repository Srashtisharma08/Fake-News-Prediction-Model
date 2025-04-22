document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const urlForm = document.getElementById("urlForm");
  const urlInput = document.getElementById("urlInput");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const errorMessage = document.getElementById("errorMessage");
  const successMessage = document.getElementById("successMessage");
  const verifyButton = document.querySelector('button:contains("Verify")');
  const realButton = document.querySelector('button:contains("Real")');
  const fakeButton = document.querySelector('button:contains("Fake")');

  // State
  let isLoading = false;
  let currentAnalysis = null;

  // URL validation
  function isValidUrl(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  // Show/hide loading state
  function setLoading(loading) {
    isLoading = loading;
    loadingSpinner.classList.toggle("hidden", !loading);
    urlInput.disabled = loading;
    verifyButton.disabled = loading;
  }

  // Show message
  function showMessage(element, message) {
    element.textContent = message;
    element.classList.remove("hidden");
    element.classList.add("message", "slide-in");
    setTimeout(() => {
      element.classList.add("hidden");
      element.classList.remove("message", "slide-in");
    }, 5000);
  }

  // Mock API call - Replace with actual API integration
  async function analyzeUrl(url) {
    setLoading(true);

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Mock response - Replace with actual API call
      const response = {
        isReal: Math.random() > 0.5,
        confidence: Math.floor(Math.random() * 30) + 70,
        analysis: {
          source: "Verified",
          date: new Date().toISOString(),
          category: "News",
        },
      };

      setLoading(false);
      return response;
    } catch (error) {
      setLoading(false);
      throw new Error("Failed to analyze URL");
    }
  }

  // Form submission
  urlForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const url = urlInput.value.trim();

    if (!isValidUrl(url)) {
      showMessage(errorMessage, "Please enter a valid URL");
      return;
    }

    try {
      const result = await analyzeUrl(url);
      currentAnalysis = result;

      // Update UI based on result
      if (result.isReal) {
        realButton.classList.add("bg-blue-700");
        fakeButton.classList.remove("bg-red-700");
        showMessage(
          successMessage,
          `Article appears to be real (${result.confidence}% confidence)`,
        );
      } else {
        fakeButton.classList.add("bg-red-700");
        realButton.classList.remove("bg-blue-700");
        showMessage(
          errorMessage,
          `Article appears to be fake (${result.confidence}% confidence)`,
        );
      }
    } catch (error) {
      showMessage(errorMessage, error.message);
    }
  });

  // Button interactions
  document.querySelectorAll("button").forEach((button) => {
    button.addEventListener("mouseenter", () => {
      button.classList.add("button-hover");
    });

    button.addEventListener("mouseleave", () => {
      button.classList.remove("button-hover");
    });
  });

  // Smooth scroll for navigation links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      e.preventDefault();
      const element = document.querySelector(anchor.getAttribute("href"));
      if (element) {
        element.scrollIntoView({
          behavior: "smooth",
        });
      }
    });
  });

  // Error boundary
  window.addEventListener("error", (event) => {
    console.error("Application error:", event.error);
    showMessage(
      errorMessage,
      "An unexpected error occurred. Please try again.",
    );
  });

  // Initialize tooltips
  const tooltipElements = document.querySelectorAll("[data-tooltip]");
  tooltipElements.forEach((element) => {
    element.addEventListener("mouseenter", (e) => {
      const tooltip = document.createElement("div");
      tooltip.className = "tooltip fade-in";
      tooltip.textContent = element.dataset.tooltip;
      document.body.appendChild(tooltip);

      const rect = element.getBoundingClientRect();
      tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`;
      tooltip.style.left = `${rect.left + (rect.width - tooltip.offsetWidth) / 2}px`;
    });

    element.addEventListener("mouseleave", () => {
      const tooltip = document.querySelector(".tooltip");
      if (tooltip) {
        tooltip.remove();
      }
    });
  });

  // Handle hero buttons
  const heroButtons = document.querySelector(".hero-buttons");
  if (heroButtons) {
    const verifyButton = heroButtons.querySelector(".verify-button");
    if (verifyButton) {
      verifyButton.addEventListener("click", () => {
        const urlSection = document.querySelector(".url-input-section");
        if (urlSection) {
          urlSection.scrollIntoView({ behavior: "smooth" });
          urlInput.focus();
        }
      });
    }
  }
});

// Function to handle form submissions with client-side validation
document.addEventListener("DOMContentLoaded", function () {
  // Add User Form Validation
  const addUserForm = document.querySelector('form[action="/add"]');
  if (addUserForm) {
    addUserForm.addEventListener("submit", function (event) {
      const username = addUserForm.querySelector('[name="newusername"]').value;
      const userId = addUserForm.querySelector('[name="newuserid"]').value;

      // Validate that both fields are filled
      if (username === "" || userId === "") {
        event.preventDefault(); // Prevent form submission
        alert("Please fill in both the username and user ID.");
      } else if (isNaN(userId) || userId <= 0) {
        event.preventDefault();
        alert("Please enter a valid positive number for User ID.");
      }
    });
  }

  // Login Form Validation
  const loginForm = document.querySelector('form[action="/login"]');
  if (loginForm) {
    loginForm.addEventListener("submit", function (event) {
      const username = loginForm.querySelector('[name="username"]').value;
      const password = loginForm.querySelector('[name="password"]').value;

      // Validate that both fields are filled
      if (username === "" || password === "") {
        event.preventDefault();
        alert("Please fill in both username and password.");
      }
    });
  }

  // Dynamic Attendance Table Filtering (Optional)
  const filterButton = document.querySelector("#filter-attendance");
  if (filterButton) {
    filterButton.addEventListener("click", function () {
      const dateInput = document.querySelector("#attendance_date").value;
      if (!dateInput) {
        alert("Please select a date to filter attendance.");
        return;
      }

      // Reload the page with the filtered date
      window.location.href = `/reports?date=${dateInput}`;
    });
  }

  // Show/Hide Delete Confirmation
  const deleteUserButtons = document.querySelectorAll(".delete-user-btn");
  deleteUserButtons.forEach(function (button) {
    button.addEventListener("click", function (event) {
      const userName = button.getAttribute("data-username");
      const confirmed = confirm(
        `Are you sure you want to delete user: ${userName}?`
      );
      if (!confirmed) {
        event.preventDefault();
      }
    });
  });

  // Real-time User Info (Preview) in Add User Form
  const usernameInput = document.querySelector('[name="newusername"]');
  const userIdInput = document.querySelector('[name="newuserid"]');
  const userPreview = document.querySelector(".user-preview");

  if (usernameInput && userIdInput && userPreview) {
    usernameInput.addEventListener("input", updateUserPreview);
    userIdInput.addEventListener("input", updateUserPreview);

    function updateUserPreview() {
      const username = usernameInput.value;
      const userId = userIdInput.value;
      userPreview.textContent = `Preview: ${username} (ID: ${userId})`;
    }
  }

  // Real-time Search in Attendance Report
  const searchInput = document.querySelector("#search-attendance");
  const attendanceTable = document.querySelector(".attendance-table tbody");

  if (searchInput && attendanceTable) {
    searchInput.addEventListener("input", function () {
      const query = searchInput.value.toLowerCase();
      const rows = attendanceTable.querySelectorAll("tr");
      rows.forEach(function (row) {
        const nameCell = row.querySelector("td:nth-child(2)");
        const rollCell = row.querySelector("td:nth-child(3)");

        const nameText = nameCell ? nameCell.textContent.toLowerCase() : "";
        const rollText = rollCell ? rollCell.textContent.toLowerCase() : "";

        if (nameText.includes(query) || rollText.includes(query)) {
          row.style.display = "";
        } else {
          row.style.display = "none";
        }
      });
    });
  }

  // Attendance Modal (Optional for better UX)
  const attendanceModal = document.querySelector("#attendanceModal");
  const startAttendanceButton = document.querySelector(
    "#start-attendance-button"
  );

  if (startAttendanceButton && attendanceModal) {
    startAttendanceButton.addEventListener("click", function () {
      attendanceModal.style.display = "block"; // Show modal when the button is clicked
    });

    const closeModalButton = attendanceModal.querySelector(".close");
    closeModalButton.addEventListener("click", function () {
      attendanceModal.style.display = "none"; // Hide modal when close button is clicked
    });
  }

  // Closing Modal when clicked outside
  window.addEventListener("click", function (event) {
    if (event.target === attendanceModal) {
      attendanceModal.style.display = "none"; // Hide modal when clicked outside
    }
  });
});
